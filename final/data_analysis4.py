# #!/usr/bin/env python3
# import argparse
# import json
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import matplotlib.pyplot as plt

# # A4 size in inches (landscape)
# FIG_SIZE_A4 = (11.69, 8.27)

# MODEL_NAME_MAP = {
#     "cnn_model": "CNN (CLS)",
#     "cnn_reg_model": "CNN (REG)",
#     "transformer_cls_model": "Transf. (CLS)",
#     "transformer_reg_model": "Transf. (REG)",
#     "rasp_model": "RASP (COMP)",
#     "trainable_rasp": "RASP (TRAIN)",
# }


# def load_json(path: Path) -> Dict:
#     try:
#         text = path.read_text(encoding="utf-8")
#     except FileNotFoundError:
#         raise FileNotFoundError(f"JSON file not found: {path}")
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError as exc:
#         raise ValueError(f"Invalid JSON in {path}: {exc}")


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


# def plot_all(metrics: Dict) -> None:
#     plt.rcParams.update({
#         'font.size': 20,
#         'axes.titlesize': 20,
#         'axes.labelsize': 20,
#         'xtick.labelsize': 14,
#         'ytick.labelsize': 14,
#     })
#     raw_models, metrics = prepare_metrics(metrics)

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

#     plt.figure(figsize=FIG_SIZE_A4)
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
    
#     for model in model_names:
#         acc = metrics.get(f"acc_{model}", [])
#         start, end = get_epoch_window(model)
        
#         if not acc:
#             continue
            
#         # Handle epoch window slicing
#         if end is None:
#             sliced_acc = acc[start:]
#             epochs = range(1000)
#         else:
#             sliced_acc = acc[start:end]
#             epochs = range(1000)
        
#         print(len(epochs))
#         print(len(sliced_acc))
#         sliced_acc = sliced_acc[:1000]
#         label = MODEL_NAME_MAP.get(model, model)
#         plt.plot(epochs, sliced_acc, label=label, linewidth=1.5)

#     plt.title(f"Accuracy Comparison Across Models (Run: 4)")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def main():
#     parser = argparse.ArgumentParser(description="Plot model training metrics.")
#     parser.add_argument(
#         "json_path",
#         type=Path,
#         nargs="?",
#         default=Path("run4_full.json"),
#         help="Path to JSON metrics file",
#     )
#     args = parser.parse_args()

#     metrics = load_json(args.json_path)
#     plot_all(metrics)


# if __name__ == "__main__":
#     main()

MODEL_COLORS = {
    "cnn_model": '#d53e4f',        # Muted blue
    "cnn_reg_model": '#fc8d59',    # Safety orange
    "transformer_cls_model": '#fee08b',  # Cooked asparagus green
    "transformer_reg_model": '#e6f598',  # Brick red
    "rasp_model": '#99d594',       # Muted purple
    "trainable_rasp": '#3288bd',   # Chestnut brown
}


#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# A4 size in inches (landscape)
FIG_SIZE_A4 = (11.69, 8.27)

MODEL_NAME_MAP = {
    "cnn_model": "CNN (CLS)",
    "cnn_reg_model": "CNN (REG)",
    "transformer_cls_model": "Transf. (CLS)",
    "transformer_reg_model": "Transf. (REG)",
    "rasp_model": "RASP (COMP)",
    "trainable_rasp": "RASP (TRAIN)",
}


def load_json(path: Path) -> Dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}")


def prepare_metrics(metrics: Dict) -> Tuple[List[str], Dict]:
    def split_interleaved(src: str, a: str, b: str):
        if src in metrics:
            vals = metrics.pop(src)
            metrics[a], metrics[b] = vals[::2], vals[1::2]

    split_interleaved("acc_cnn_reg_model", "acc_cnn_model", "acc_cnn_reg_model")
    split_interleaved(
        "acc_transformer_reg_model",
        "acc_transformer_cls_model",
        "acc_transformer_reg_model",
    )

    loss_keys = [k for k in metrics if k.startswith("loss_")]
    model_names = [k.removeprefix("loss_") for k in loss_keys]
    return model_names, metrics


def get_epoch_window(model: str) -> Tuple[Optional[int], Optional[int]]:
    windows = {
        "cnn_model": (0, 1000),
        "cnn_reg_model": (1000, 2000),
        "transformer_cls_model": (2000, 3000),
        "transformer_reg_model": (3000, None),
    }
    return windows.get(model, (None, None))


def plot_all(metrics: Dict) -> None:
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })
    raw_models, metrics = prepare_metrics(metrics)

    desired = [
        "cnn_model",
        "cnn_reg_model",
        "transformer_cls_model",
        "transformer_reg_model",
        "rasp_model",
        "trainable_rasp",
    ]
    model_names = [m for m in desired if m in raw_models]
    if not model_names:
        print("No recognized models found.")
        return

    plt.figure(figsize=FIG_SIZE_A4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    for model in model_names:
        acc = metrics.get(f"acc_{model}", [])
        start, end = get_epoch_window(model)
        
        if not acc:
            continue
            
        # Handle epoch window slicing
        if end is None:
            sliced_acc = acc[start:]
            epochs = range(1000)
        else:
            sliced_acc = acc[start:end]
            epochs = range(1000)
        
        sliced_acc = sliced_acc[:1000]
        label = MODEL_NAME_MAP.get(model, model)
        plt.plot(epochs, sliced_acc, label=label, linewidth=3)  # Increased linewidth

    # Add vertical line at epoch 100
    plt.axvline(x=95, color='red', linestyle='--', linewidth=3)
    
    plt.title(f"Accuracy Comparison Across Models (Run: 4)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_token(tok_his: Dict) -> None:
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    # Extract and sort epochs numerically
    epochs = sorted(map(int, tok_his.keys()))
    if not epochs:
        print("No token history data found.")
        return

    # Determine number of token positions from the first epoch's data
    num_positions = len(tok_his[str(epochs[0])])
    
    # Prepare data for each token position
    position_data = [[] for _ in range(num_positions)]
    for epoch in epochs:
        values = tok_his[str(epoch)]
        for i in range(num_positions):
            position_data[i].append(values[i])

    # Create plot
    plt.figure(figsize=FIG_SIZE_A4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    # Use a colormap with enough distinct colors
    cmap = plt.get_cmap('tab10')
    for i in range(num_positions):
        color = cmap(i % 10)  # Reuse colors if more than 10 positions
        plt.plot(
            epochs,
            position_data[i],
            label=f"Position {i+1}",
            color=color,
            linewidth=3
        )

    plt.title("Token Position Accuracy Over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot model training metrics.")
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=Path("run7_full.json"),
        help="Path to JSON metrics file",
    )
    args = parser.parse_args()

    metrics = load_json(args.json_path)
    print(metrics['tok_his'])
    plot_all(metrics)
    plot_token(metrics['tok_his'])


if __name__ == "__main__":
    main()