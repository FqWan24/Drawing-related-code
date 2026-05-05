import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = "logs/cub"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

METHODS = {
    "vanilla": "Vanilla",
    "dlora": r"SR$^2$-LoRA",
}

OPTIMIZER = "adam"

BACKBONE_ORDER = [
    "small_1k",
    "base_1k",
    "large_1k",
    "small_ink21k",
    "large_ink21k",
]


def parse_log(log_path):
    """只读最后 5 行，提取 total 和 avg"""
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-5:]

    total = None
    avg = None

    for line in lines:
        if "CNN:" in line:
            m = re.search(r"'total':\s*np\.float64\(([\d.]+)\)", line)
            if m:
                total = float(m.group(1))

        if "CNN top1 avg" in line:
            m = re.search(r"CNN top1 avg:\s*([\d.]+)", line)
            if m:
                avg = float(m.group(1))

    return total, avg


def aggregate_method(log_dir):
    totals = []
    avgs = []

    if not os.path.exists(log_dir):
        return None

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith(".log"):
            continue

        total, avg = parse_log(os.path.join(log_dir, filename))

        if total is not None:
            totals.append(total)
        if avg is not None:
            avgs.append(avg)

    if len(totals) == 0 or len(avgs) == 0:
        return None

    return {
        "last_mean": np.mean(totals),
        "last_std": np.std(totals, ddof=1) if len(totals) > 1 else 0.0,
        "avg_mean": np.mean(avgs),
        "avg_std": np.std(avgs, ddof=1) if len(avgs) > 1 else 0.0,
        "num_seeds": len(totals),
    }


def collect_results():
    rows = []

    for exp_name in sorted(os.listdir(ROOT_DIR)):
        exp_path = os.path.join(ROOT_DIR, exp_name)
        if not os.path.isdir(exp_path):
            continue

        backbone = exp_name.replace("10_10_sip_", "")
        row = {
            "experiment": exp_name,
            "backbone": backbone,
        }

        valid = True

        for method in METHODS:
            log_dir = os.path.join(exp_path, method, OPTIMIZER)
            stats = aggregate_method(log_dir)

            if stats is None:
                valid = False
                break

            row[f"{method}_avg_mean"] = stats["avg_mean"]
            row[f"{method}_avg_std"] = stats["avg_std"]
            row[f"{method}_last_mean"] = stats["last_mean"]
            row[f"{method}_last_std"] = stats["last_std"]
            row[f"{method}_num_seeds"] = stats["num_seeds"]

        if valid:
            rows.append(row)

    df = pd.DataFrame(rows)

    df["backbone"] = pd.Categorical(
        df["backbone"],
        categories=BACKBONE_ORDER,
        ordered=True
    )

    df = df.sort_values("backbone")
    return df


def beautify_backbone_name(name):
    mapping = {
        "small_1k": "Small-1K",
        "base_1k": "Base-1K",
        "large_1k": "Large-1K",
        "small_ink21k": "Small-21K",
        "large_ink21k": "Large-21K",
    }
    return mapping.get(name, name)


def plot_metric(df, metric="avg"):
    assert metric in ["avg", "last"]

    x = np.arange(len(df))
    width = 0.34

    vanilla_mean = df[f"vanilla_{metric}_mean"].values
    vanilla_std = df[f"vanilla_{metric}_std"].values

    dlora_mean = df[f"dlora_{metric}_mean"].values
    dlora_std = df[f"dlora_{metric}_std"].values

    gains = dlora_mean - vanilla_mean

    fig, ax = plt.subplots(figsize=(8, 6))

    red = "#ef4b3f"
    teal = "#5abfaf"

    bars_dlora = ax.bar(
        x - width / 2,
        dlora_mean,
        width,
        yerr=dlora_std,
        capsize=4,
        label=r"SR$^2$-LoRA",
        facecolor="white",
        edgecolor=red,
        linewidth=2.0,
        hatch="**",
        error_kw=dict(ecolor=red, lw=1.8, capsize=4),
    )

    bars_vanilla = ax.bar(
        x + width / 2,
        vanilla_mean,
        width,
        yerr=vanilla_std,
        capsize=4,
        label="Vanilla",
        facecolor="white",
        edgecolor=teal,
        linewidth=2.0,
        hatch="..",
        error_kw=dict(ecolor=teal, lw=1.8, capsize=4),
    )

    # 画提升箭头
    for i in range(len(df)):
        x_arrow = x[i] + width / 2
        y0 = vanilla_mean[i]
        y1 = dlora_mean[i]

        ax.annotate(
            "",
            xy=(x_arrow, y1),
            xytext=(x_arrow, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=red,
                lw=1.6,
                linestyle="--",
            ),
        )

        ax.text(
            x_arrow-0.15,
            (y0 + y1) / 2,
            f"+{gains[i]:.2f}",
            color=red,
            fontsize=13,
            fontweight="bold",
            va="center",
        )

    xticklabels = [beautify_backbone_name(str(b)) for b in df["backbone"]]

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=30, rotation=10, ha="right")

    if metric == "avg":
        ax.set_ylabel("Avg Accuracy (%)", fontsize=28)
        # ax.set_title("Backbone Effect on Avg Accuracy", fontsize=22)
        save_name = "backbone_avg_bar"
    else:
        ax.set_ylabel("Last Accuracy (%)", fontsize=28)
        # ax.set_title("Backbone Effect on Last Accuracy", fontsize=22)
        save_name = "backbone_last_bar"

    # ax.set_xlabel("Backbone", fontsize=20)

    ymin = min(vanilla_mean.min(), dlora_mean.min()) - 2.0
    ymax = max(vanilla_mean.max(), dlora_mean.max()) + 2.5
    ax.set_ylim(ymin, ymax)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=20)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        frameon=False,
        fontsize=20,
    )

    # 去掉上边和右边
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 保留左和下（论文常用风格）
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)


    plt.tight_layout()

    png_path = os.path.join(SAVE_DIR, f"{save_name}.png")
    pdf_path = os.path.join(SAVE_DIR, f"{save_name}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.show()


def main():
    df = collect_results()

    if df.empty:
        print("No valid results found.")
        return

    df["avg_gain"] = df["dlora_avg_mean"] - df["vanilla_avg_mean"]
    df["last_gain"] = df["dlora_last_mean"] - df["vanilla_last_mean"]

    print("\n===== Results =====")
    print(df[
        [
            "backbone",
            "vanilla_avg_mean",
            "dlora_avg_mean",
            "avg_gain",
            "vanilla_last_mean",
            "dlora_last_mean",
            "last_gain",
        ]
    ].to_string(index=False))

    csv_path = os.path.join(SAVE_DIR, "backbone_vanilla_vs_sr2lora.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    plot_metric(df, metric="avg")
    plot_metric(df, metric="last")


if __name__ == "__main__":
    main()