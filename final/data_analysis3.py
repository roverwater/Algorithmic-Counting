# #!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# # A4 size in inches (landscape)
# FIG_SIZE_A4 = (11.69, 8.27)

# # -- Map your raw model keys → pretty labels here: --
# MODEL_NAME_MAP = {
#     "cnn_model":         "CNN (CLS)",
#     "cnn_reg_model":     "CNN (REG)",
#     "transformer_cls_model": "Transf. (CLS)",
#     "transformer_reg_model": "Transf. (REG)",
#     "rasp_model":        "RASP (COMP)",
#     "trainable_rasp":    "RASP (TRAIN)",
# }


def load_json(path: Path) -> Dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}")


# def prepare_metrics(metrics: Dict) -> Tuple[List[str], Dict]:
#     def split_interleaved(src: str, a: str, b: str):
#         if src in metrics:
#             vals = metrics.pop(src)
#             metrics[a], metrics[b] = vals[::2], vals[1::2]

#     split_interleaved("acc_cnn_reg_model", "acc_cnn_model", "acc_cnn_reg_model")
#     split_interleaved(
#         "acc_transformer_reg_model",
#         "acc_transformer_cls_model",
#         "acc_transformer_reg_model",
#     )

#     loss_keys = [k for k in metrics if k.startswith("loss_")]
#     model_names = [k.removeprefix("loss_") for k in loss_keys]
#     return model_names, metrics


# def get_epoch_window(model: str) -> Tuple[Optional[int], Optional[int]]:
#     windows = {
#         "cnn_model": (0, 1000),
#         "cnn_reg_model": (1000, 2000),
#         "transformer_cls_model": (2000, 3000),
#         "transformer_reg_model": (3000, None),
#     }
#     return windows.get(model, (None, None))


# def accuracy_key(model: str) -> str:
#     if model == "rasp_model":
#         return "original"
#     if model == "trainable_rasp":
#         return "model_trainable"
#     return model.removesuffix("_model")


# def plot_model_row(
#     row: int,
#     model: str,
#     metrics: Dict,
#     fig: plt.Figure,
#     gs: GridSpec,
#     total_models: int,
# ) -> None:
#     axL = fig.add_subplot(gs[row, 0])
#     axA = fig.add_subplot(gs[row, 1])
#     axB = fig.add_subplot(gs[row, 2])

#     # LOSS
#     loss = metrics.get(f"loss_{model}", [])
#     axL.plot(loss)
#     pretty = MODEL_NAME_MAP.get(model, model)
#     axL.set_ylabel(pretty, rotation=0, labelpad=40, va="center")
#     if row == total_models:
#         axL.set_xlabel("Epoch")

#     # ACCURACY
#     acc = metrics.get(f"acc_{model}", [])
#     start, end = get_epoch_window(model)
#     axA.plot(acc[start:end] if end else acc[start:])
#     if row == total_models:
#         axA.set_xlabel("Epoch")

#     # COUNT-WISE ACC
#     if "accuracy_res" in metrics:
#         key = accuracy_key(model)
#         res = metrics["accuracy_res"].get(key, [])
#         counts = metrics.get("count_freq", list(range(len(res))))
#         if len(counts) != len(res):
#             counts = list(range(len(res)))
#         axB.bar(counts, res, color="orange", edgecolor="black", linewidth=0.5)
#         axB.set_ylim(0, 100)
#         if row == total_models:
#             axB.set_xlabel("Count")


# def plot_count_histograms(
#     row: int,
#     metrics: Dict,
#     fig: plt.Figure,
#     gs: GridSpec,
# ) -> None:
#     ax_train = fig.add_subplot(gs[row, 0])
#     ax_test  = fig.add_subplot(gs[row, 1])
#     # leave gs[row,2] empty

#     train = metrics.get("train_count_freq", [])
#     test  = metrics.get("test_count_freq", [])
#     if not (train and test):
#         return

#     all_vals = train + test
#     bins = np.arange(min(all_vals), max(all_vals) + 2) - 0.5

#     ax_train.hist(train, bins=bins, edgecolor="black", alpha=0.7)
#     ax_test.hist(test,  bins=bins, edgecolor="black", alpha=0.7, color="orange")

#     ax_train.set_title("Train Counts")
#     ax_test.set_title("Test Counts")


# def plot_all(metrics: Dict) -> None:
#     raw_models, metrics = prepare_metrics(metrics)

#     # enforce your desired order & drop any missing
#     desired = [
#         "cnn_model",
#         "cnn_reg_model",
#         "transformer_cls_model",
#         "transformer_reg_model",
#         "rasp_model",
#         "trainable_rasp",
#     ]
#     model_names = [m for m in desired if m in raw_models]
#     if not model_names:
#         print("No recognized models found.")
#         return

#     total = len(model_names)
#     # +1 for histograms row at top
#     n_rows = total + 1

#     fig = plt.figure(figsize=FIG_SIZE_A4)
#     gs  = GridSpec(n_rows, 3, figure=fig, wspace=0.4, hspace=0.6)

#     # 1) Histogram row at index 0
#     plot_count_histograms(0, metrics, fig, gs)

#     # 2) Column headers just above row 1 (the first model)
#     headers = ["Loss", "Accuracy", "Count-wise Accuracy"]
#     xpos    = [0.22, 0.51, 0.80]
#     # y=0.86 sits below the histograms, above the model rows
#     for x, txt in zip(xpos, headers):
#         fig.text(x, 0.02, txt, ha="center", va="center", fontsize=12)

#     # 3) Model rows at 1..total
#     for idx, mdl in enumerate(model_names, start=1):
#         plot_model_row(idx, mdl, metrics, fig, gs, total)

#     # 4) Super‐title a little below the very top
#     seed = metrics.get("seed", "N/A")
#     fig.suptitle(f"Training Metrics (Seed: {seed})", y=0.95, fontsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#     plt.show()


# import json
# from pathlib import Path
# import matplotlib.pyplot as plt

# def plot_count_freq(json_path: Path = Path("run7_full.json")):
#     # — Option 1: globally increase all font sizes —
#     plt.rcParams.update({
#         'font.size': 14,        # default text
#         'axes.titlesize': 16,   # axes title
#         'axes.labelsize': 14,   # x/y labels
#         'xtick.labelsize': 12,  # x‐tick labels
#         'ytick.labelsize': 12,  # y‐tick labels
#     })

#     # Load metrics from JSON file
#     with open(json_path, 'r') as f:
#         metrics = json.load(f)
#         print(metrics.keys())
    
#     # Extract count_freq
#     count_freq = metrics.get('count_freq', [])
#     if not count_freq:
#         print("No 'count_freq' data found in the JSON file.")
#         return
    
#     # Determine integer bins
#     bins = range(min(count_freq), max(count_freq) + 2)
    
#     # Create histogram
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.hist(count_freq, bins=bins, edgecolor='black', alpha=0.7)
    
#     # — Option 2: per‐call font overrides (if you prefer not to touch rcParams) —
#     # ax.set_title("Count Distribution", fontsize=16)
#     # ax.set_xlabel("Object count",      fontsize=14)
#     # ax.set_ylabel("Number of samples", fontsize=14)
#     # ax.tick_params(axis='both', which='major', labelsize=12)
    
#     ax.set_title("Count Distribution")
#     ax.set_xlabel("Object count")
#     ax.set_ylabel("Number of samples")
#     ax.set_xticks(bins)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Plot count_freq histogram.")
#     parser.add_argument(
#         "json_path",
#         type=Path,
#         nargs="?",
#         default=Path("run1_full.json"),
#         help="Path to the JSON metrics file."
#     )
#     args = parser.parse_args()
#     plot_count_freq(args.json_path)

import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_count_freq(metrics):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    count_freq = metrics.get('count_freq', [])
    if not count_freq:
        print("No 'count_freq' data found.")
        return

    bins = range(min(count_freq), max(count_freq) + 2)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(count_freq, bins=bins, edgecolor='black', alpha=0.7)
    
    ax.set_title("Count Distribution")
    ax.set_xlabel("Object count")
    ax.set_ylabel("Number of samples")
    ax.set_xticks(bins)
    plt.tight_layout()
    plt.show()

def plot_accuracies(metrics):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # Extract accuracy metrics
    acc_keys = [k for k in metrics if k.startswith('acc_')]
    if not acc_keys:
        print("No accuracy metrics found.")
        return

    # Process values with empty list handling
    valid_data = []
    for k in acc_keys:
        val = metrics[k]
        if isinstance(val, list):
            if len(val) == 0:
                print(f"Warning: {k} is empty, skipping")
                continue
            print(f"Warning: {k} is a list, using first element")
            val = val[0]
        
        try:
            valid_data.append((
                k.replace('acc_', '').replace('_', ' ').title(),
                float(val) * 100
            ))
        except ValueError:
            print(f"Warning: Could not convert value for {k}, skipping")

    if not valid_data:
        print("No valid accuracy data to plot")
        return

    # Unpack valid data
    model_names, acc_values = zip(*valid_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, acc_values, color='skyblue')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    ax.set_title("Model Accuracies")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot metrics from JSON file.")
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=Path("run1_full.json"),
        help="Path to the JSON metrics file."
    )
    args = parser.parse_args()
    
    # Load metrics once and share between plots
    with open(args.json_path, 'r') as f:
        metrics = json.load(f)
    
    plot_count_freq(metrics)
    plot_accuracies(metrics)


