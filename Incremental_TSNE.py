import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import os
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_dlora import SiNet
from models.vit_dlora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import math
    
class DLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.args = args
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.disloss = args["disloss"]
        self.loss_rel_history = []

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []
        
        self.task1_ref_features = None
        self.task1_ref_relations = None
        self.task1_ref_margins = None
        self.task1_ref_targets = None

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_load(self, data_manager, logfilename):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        # self.state_spare(state, self._cur_task)
        state = torch.load(os.path.join(logfilename, f"task_{self._cur_task}.pth"))
        self._network.load_state_dict(state)
        self._network.to(self._device)
        
        self._network.eval()
        self.plot_task1_decision_boundary(
            save_path=f"./tsne_task1_boundary/task1_boundary_task_{self._cur_task + 1}.pdf",
            first_task_size=10,
            max_per_class=80,
        )
        
    @torch.no_grad()
    def plot_task1_decision_boundary(
        self,
        save_path="./task1_boundary.pdf",
        first_task_size=10,
        max_per_class=80,
        perplexity=30,
        knn_k=15,
    ):
        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.neighbors import KNeighborsClassifier

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._network.eval()

        selected_classes = list(range(first_task_size))
        label_to_idx = {c: i for i, c in enumerate(selected_classes)}

        features_list = []
        labels_list = []
        class_count = {c: 0 for c in selected_classes}

        for _, inputs, targets in self.test_loader:
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for c in selected_classes:
                mask |= (targets == c)

            if mask.sum() == 0:
                continue

            inputs = inputs[mask].to(self._device)
            targets = targets[mask].cpu()

            results = self._network(inputs)

            if isinstance(results, dict):
                if "features" in results:
                    features = results["features"]
                elif "feature" in results:
                    features = results["feature"]
                else:
                    raise KeyError("Cannot find feature key in model output.")
            else:
                raise TypeError("Model output should be a dict containing features.")

            features = features.detach().cpu()

            for feat, label in zip(features, targets):
                label_int = int(label.item())

                if label_int not in class_count:
                    continue

                if class_count[label_int] >= max_per_class:
                    continue

                features_list.append(feat)
                labels_list.append(label_to_idx[label_int])
                class_count[label_int] += 1

        if len(features_list) == 0:
            print("No Task-1 samples found.")
            return

        features = torch.stack(features_list, dim=0).numpy()
        labels = np.array(labels_list)

        n_samples = features.shape[0]
        effective_perplexity = min(perplexity, max(5, (n_samples - 1) // 3))

        features_2d = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate="auto",
            init="pca",
            random_state=0,
        ).fit_transform(features)

        clf = KNeighborsClassifier(n_neighbors=min(knn_k, len(labels)))
        clf.fit(features_2d, labels)

        x_min, x_max = features_2d[:, 0].min() - 2.0, features_2d[:, 0].max() + 2.0
        y_min, y_max = features_2d[:, 1].min() - 2.0, features_2d[:, 1].max() + 2.0

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 500),
            np.linspace(y_min, y_max, 500),
        )

        zz = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.figure(figsize=(4.6, 4.0))

        levels = np.arange(first_task_size + 1) - 0.5

        plt.contourf(
            xx,
            yy,
            zz,
            levels=levels,
            cmap="tab10",
            alpha=0.32,
        )

        plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap="tab10",
            vmin=0,
            vmax=first_task_size - 1,
            marker="o",
            s=22,
            edgecolors="none",
            linewidths=0,
            alpha=0.9,
        )

        plt.xticks([])
        plt.yticks([])
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300,pad_inches=0)
        plt.close()

        print(f"Saved Task-1 decision boundary to {save_path}")
        
    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        
        self._train(self.train_loader, self.test_loader)
        
        self._compute_mean(data_manager)
        
        if self._cur_task > 0:
            self.classifer_align()

        

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if "classifier_pool" + "." + str(self._network.numtask - 1) + "." in name:
                param.requires_grad_(True)
            if "lora_B_k" + "." + str(self._network.numtask - 1) + "."  in name:
                param.requires_grad_(True)
            if "lora_B_v" + "." + str(self._network.numtask - 1) + "."  in name:
                param.requires_grad_(True)
            if "lora_A_k" + "." + str(self._network.numtask - 1) + "."  in name:
                param.requires_grad_(True)
            if "lora_A_v" + "." + str(self._network.numtask - 1) + "."  in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if self._cur_task==0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.lrate,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        return


    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):

            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes
                
                results = self._network(inputs)

                logits = results['logits']
                 
                loss = F.cross_entropy(logits, targets)


                if results['feature_old_list'] is not None:
                    feat_old = torch.stack(results['feature_old_list'], dim=1)
                    feat_new = torch.stack(results['feature_new_list'], dim=1)
                    rel_old = torch.bmm(feat_old, feat_old.transpose(1, 2))
                    rel_new = torch.bmm(feat_new, feat_new.transpose(1, 2))
                    loss_rel = spectral_relation_loss(rel_new, rel_old)
                
                    loss = loss + loss_rel



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
  
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets//self.class_num).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]
    
    @torch.no_grad()
    def _compute_mean(self, data_manager):
        self._network.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                inputs, targets = _inputs.to(self._device), _targets.to(self._device)
                _vectors = self._network.extract_vector(inputs)
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            
            features_per_cls = vectors
            # print(features_per_cls.shape)
            self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
            # self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (
            #         torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (
                    torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-2).to(self._device)
            
    
    def classifer_align(self):
    
        from torch.distributions.multivariate_normal import MultivariateNormal
        for p in self._network.classifier_pool.parameters():
            p.requires_grad = True

        run_epochs = 20
        #  param_list = [p for n, p in self._network.fc.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [
            {'params': self._network.classifier_pool.parameters(), 'lr': 0.005, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=0.005, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)

        prog_bar = tqdm(range(run_epochs))
        task_size = self._total_classes - self._known_classes
        self._network.eval()
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            for class_idx in range(self._total_classes):
                
                mean = self.cls_mean[class_idx].to(self._device)
                cov = self.cls_cov[class_idx].to(self._device)
                m = MultivariateNormal(mean.float(), cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)

                sampled_label.extend([class_idx] * num_sampled_pcls)

            
            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                logits = self._network(inp, fc_only = True)
                loss = F.cross_entropy(logits, tgt)
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                run_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)

        logging.info(info) 
        
        
def spectral_relation_loss(rel_new, rel_old):
    B, L, _ = rel_new.shape
    rel_new_sym = (rel_new + rel_new.transpose(1, 2)) / 2
    rel_old_sym = (rel_old + rel_old.transpose(1, 2)) / 2
    evals_new = torch.linalg.eigvalsh(rel_new_sym)
    evals_old = torch.linalg.eigvalsh(rel_old_sym).detach()
    return torch.nn.functional.smooth_l1_loss(evals_new, evals_old)