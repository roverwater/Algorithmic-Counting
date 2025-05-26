#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

def plot_accuracies(metrics):
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    })

MODEL_NAME_MAP = {
    "cnn_model": "CNN (CLS)",
    "cnn_reg_model": "CNN (REG)",
    "transformer_cls_model": "Transf. (CLS)",
    "transformer_reg_model": "Transf. (REG)",
    "rasp_model": "RASP (COMP)",
    "trainable_rasp": "RASP (TRAIN)",
}

def accuracy_key(model: str) -> str:
    if model == "rasp_model":
        return "original"
    if model == "trainable_rasp":
        return "model_trainable"
    return model.removesuffix("_model")

def load_json(path: Path) -> Dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}")

def main():
    desired_models = [
        "cnn_model",
        "cnn_reg_model",
        "transformer_cls_model",
        "transformer_reg_model",
        "rasp_model",
        "trainable_rasp",
    ]

    sum_acc = {model: None for model in desired_models}
    model_counts = {model: None for model in desired_models}  # Track counts per model

    runs = [1, 4, 5, 6, 7]
    nruns = len(runs)

    for run in runs:  # Iterate over runs 1 to 7
        json_path = Path(f"run{run}_full.json")
        try:
            metrics = load_json(json_path)
        except FileNotFoundError:
            print(f"Skipping missing file: {json_path}")
            continue
        accuracy_res = metrics.get("accuracy_res", {})
        run_count_freq = metrics.get("count_freq")

        for model in desired_models:
            key = accuracy_key(model)
            model_acc = accuracy_res.get(key, [])
            if not model_acc:
                continue  # Skip if no data for this model in the current run

            # Determine counts for this model in current run
            if run_count_freq is not None:
                current_counts = run_count_freq
            else:
                current_counts = list(range(len(model_acc)))

            # Initialize sum and counts tracking for the model
            if sum_acc[model] is None:
                sum_acc[model] = [0.0] * len(model_acc)
                model_counts[model] = current_counts
            else:
                # Check if current run's counts match the tracked counts
                if current_counts != model_counts[model]:
                    print(f"Warning: {model} in run {run} has different counts, skipping")
                    continue

            # Check data length consistency
            if len(model_acc) != len(sum_acc[model]):
                print(f"Warning: {model} in run {run} has unexpected length, skipping")
                continue

            # Accumulate the accuracy values
            for i in range(len(model_acc)):
                sum_acc[model][i] += model_acc[i]

    # Calculate average accuracy for each model
    avg_acc = {}
    for model in desired_models:
        if sum_acc[model] is not None:
            avg_acc[model] = [acc / nruns for acc in sum_acc[model]]
        else:
            avg_acc[model] = []

    # Plot setup
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, model in enumerate(desired_models):
        ax = axes[i]
        model_avg = avg_acc.get(model, [])
        if not model_avg:
            continue  # Skip if no data for the model

        pretty_name = MODEL_NAME_MAP.get(model, model)
        # Use model-specific counts if available and matching, else indices
        counts = model_counts[model] if model_counts[model] is not None else []
        if counts and len(counts) == len(model_avg):
            x = counts
        else:
            x = list(range(len(model_avg)))

        ax.bar(x, model_avg, color='orange', edgecolor='black', linewidth=0.5)
        ax.set_title(pretty_name)
        ax.set_xlabel("Count")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.suptitle("Average Count-wise Accuracy Across 5 Runs", y=1.02)
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

MODEL_NAME_MAP = {
    "cnn_model": "CNN (CLS)",
    "cnn_reg_model": "CNN (REG)",
    "transformer_cls_model": "Transf. (CLS)",
    "transformer_reg_model": "Transf. (REG)",
    "rasp_model": "RASP (COMP)",
    "trainable_rasp": "RASP (TRAIN)",
}

def accuracy_key(model: str) -> str:
    if model == "rasp_model":
        return "original"
    if model == "trainable_rasp":
        return "model_trainable"
    return model.removesuffix("_model")

def load_json(path: Path) -> Dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}")

def main():
    desired_models = [
        "cnn_model",
        "cnn_reg_model",
        "transformer_cls_model",
        "transformer_reg_model",
        "rasp_model",
        "trainable_rasp",
    ]

    sum_acc = {model: None for model in desired_models}
    model_counts = {model: None for model in desired_models}

    runs = [1, 4, 5, 6, 7]
    nruns = len(runs)

    for run in runs:
        json_path = Path(f"run{run}_full.json")
        try:
            metrics = load_json(json_path)
        except FileNotFoundError:
            print(f"Skipping missing file: {json_path}")
            continue
        accuracy_res = metrics.get("accuracy_res", {})
        run_count_freq = metrics.get("count_freq")

        for model in desired_models:
            key = accuracy_key(model)
            model_acc = accuracy_res.get(key, [])
            if not model_acc:
                continue

            if run_count_freq is not None:
                current_counts = run_count_freq
            else:
                current_counts = list(range(len(model_acc)))

            if sum_acc[model] is None:
                sum_acc[model] = [0.0] * len(model_acc)
                model_counts[model] = current_counts
            else:
                if current_counts != model_counts[model]:
                    print(f"Warning: {model} in run {run} has different counts, skipping")
                    continue

            if len(model_acc) != len(sum_acc[model]):
                print(f"Warning: {model} in run {run} has unexpected length, skipping")
                continue

            for i in range(len(model_acc)):
                sum_acc[model][i] += model_acc[i]

    avg_acc = {}
    for model in desired_models:
        if sum_acc[model] is not None:
            avg_acc[model] = [acc / nruns for acc in sum_acc[model]]
        else:
            avg_acc[model] = []

    # Generate individual plots
    for model in desired_models:
        model_avg = avg_acc.get(model, [])
        if not model_avg:
            continue

        pretty_name = MODEL_NAME_MAP.get(model, model)
        counts = model_counts[model] if model_counts[model] is not None else list(range(len(model_avg)))
        if counts and len(counts) != len(model_avg):
            print(f"Warning: {model} counts don't match, using indices")
            counts = list(range(len(model_avg)))

        plt.figure(figsize=(8, 6))
        plt.bar(counts, model_avg, color='orange', edgecolor='black', linewidth=0.5)
        plt.title(pretty_name, fontsize=20)
        plt.xlabel("Count", fontsize=20)
        plt.ylabel("Average Accuracy (%)", fontsize=20)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{model}_accuracy.png", bbox_inches='tight')
        plt.close()

    # Generate combined subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, model in enumerate(desired_models):
        ax = axes[i]
        model_avg = avg_acc.get(model, [])
        if not model_avg:
            continue

        pretty_name = MODEL_NAME_MAP.get(model, model)
        counts = model_counts[model] if model_counts[model] is not None else list(range(len(model_avg)))
        if counts and len(counts) != len(model_avg):
            counts = list(range(len(model_avg)))

        ax.bar(counts, model_avg, color='orange', edgecolor='black', linewidth=0.5)
        ax.set_title(pretty_name)
        ax.set_xlabel("Count", fontsize=20)
        ax.set_ylabel("Average Accuracy (%)", fontsize=20)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.suptitle("Average Count-wise Accuracy Across 5 Runs", y=1.02)
    plt.show()

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import json
# from pathlib import Path
# from typing import Dict, List

# import matplotlib.pyplot as plt

# MODEL_NAME_MAP = {
#     "cnn_model": "CNN (CLS)",
#     "cnn_reg_model": "CNN (REG)",
#     "transformer_cls_model": "Transf. (CLS)",
#     "transformer_reg_model": "Transf. (REG)",
#     "rasp_model": "RASP (COMP)",
#     "trainable_rasp": "RASP (TRAIN)",
# }

# def accuracy_key(model: str) -> str:
#     if model == "rasp_model":
#         return "original"
#     if model == "trainable_rasp":
#         return "model_trainable"
#     return model.removesuffix("_model")

# def load_json(path: Path) -> Dict:
#     try:
#         text = path.read_text(encoding="utf-8")
#     except FileNotFoundError:
#         raise FileNotFoundError(f"JSON file not found: {path}")
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError as exc:
#         raise ValueError(f"Invalid JSON in {path}: {exc}")

# def main():
#     desired_models = [
#         "cnn_model",
#         "cnn_reg_model",
#         "transformer_cls_model",
#         "transformer_reg_model",
#         "rasp_model",
#         "trainable_rasp",
#     ]

#     sum_acc = {model: None for model in desired_models}
#     model_counts = {model: None for model in desired_models}

#     runs = [1, 4, 5, 6, 7]
#     nruns = len(runs)

#     # Load dataset distribution from run7
#     try:
#         run7_metrics = load_json(Path("run7_full.json"))
#         count_freq = run7_metrics.get("count_freq", [])
#     except Exception as e:
#         print(f"Error loading dataset distribution: {e}")
#         count_freq = []

#     for run in runs:
#         json_path = Path(f"run{run}_full.json")
#         try:
#             metrics = load_json(json_path)
#         except FileNotFoundError:
#             print(f"Skipping missing file: {json_path}")
#             continue
#         accuracy_res = metrics.get("accuracy_res", {})
#         run_count_freq = metrics.get("count_freq")

#         for model in desired_models:
#             key = accuracy_key(model)
#             model_acc = accuracy_res.get(key, [])
#             if not model_acc:
#                 continue

#             if run_count_freq is not None:
#                 current_counts = run_count_freq
#             else:
#                 current_counts = list(range(len(model_acc)))

#             if sum_acc[model] is None:
#                 sum_acc[model] = [0.0] * len(model_acc)
#                 model_counts[model] = current_counts
#             else:
#                 if current_counts != model_counts[model]:
#                     print(f"Warning: {model} in run {run} has different counts, skipping")
#                     continue

#             if len(model_acc) != len(sum_acc[model]):
#                 print(f"Warning: {model} in run {run} has unexpected length, skipping")
#                 continue

#             for i in range(len(model_acc)):
#                 sum_acc[model][i] += model_acc[i]

#     avg_acc = {}
#     for model in desired_models:
#         if sum_acc[model] is not None:
#             avg_acc[model] = [acc / nruns for acc in sum_acc[model]]
#         else:
#             avg_acc[model] = []

#     # Generate individual plots with histograms
#     for model in desired_models:
#         model_avg = avg_acc.get(model, [])
#         if not model_avg:
#             continue

#         pretty_name = MODEL_NAME_MAP.get(model, model)
#         counts = model_counts[model] if model_counts[model] is not None else list(range(len(model_avg)))
#         if counts and len(counts) != len(model_avg):
#             counts = list(range(len(model_avg)))

#         plt.rcParams.update({
#             'font.size': 12,
#             'axes.titlesize': 14,
#             'axes.labelsize': 12,
#             'xtick.labelsize': 10,
#             'ytick.labelsize': 10,
#         })

#         fig, (ax1, ax2) = plt.subplots(
#             2, 1, 
#             figsize=(8, 7),
#             gridspec_kw={'height_ratios': [2, 1]},
#             sharex=True
#         )
        
#         # Plot accuracy
#         ax1.bar(counts, model_avg, color='orange', edgecolor='black', alpha=0.8)
#         ax1.set_title(f"{pretty_name} Accuracy", pad=15)
#         ax1.set_ylabel("Accuracy (%)")
#         ax1.set_ylim(0, 100)
#         ax1.grid(True, linestyle=':', alpha=0.7)
        
#         # Plot dataset distribution
#         if count_freq:
#             bins = range(min(count_freq), max(count_freq) + 2)
#             ax2.hist(
#                 count_freq, bins=bins,
#                 color='teal', edgecolor='black',
#                 alpha=0.8, density=False
#             )
#             ax2.set_title("Dataset Distribution", pad=15)
#             ax2.set_xlabel("Object Count")
#             ax2.set_ylabel("Samples")
#             ax2.set_xticks(bins)
#             ax2.grid(True, linestyle=':', alpha=0.7)
#         else:
#             ax2.remove()
            

#         plt.tight_layout()
#         plt.savefig(f"{model}_accuracy_with_distribution.png", dpi=150, bbox_inches='tight')
#         plt.close()

#     # Generate combined subplots (original version)
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()

#     for i, model in enumerate(desired_models):
#         ax = axes[i]
#         model_avg = avg_acc.get(model, [])
#         if not model_avg:
#             continue

#         pretty_name = MODEL_NAME_MAP.get(model, model)
#         counts = model_counts[model] if model_counts[model] is not None else list(range(len(model_avg)))
#         if counts and len(counts) != len(model_avg):
#             counts = list(range(len(model_avg)))

#         ax.bar(counts, model_avg, color='orange', edgecolor='black', linewidth=0.5)
#         ax.set_title(pretty_name)
#         ax.set_xlabel("Count")
#         ax.set_ylabel("Average Accuracy (%)")
#         ax.set_ylim(0, 100)
#         ax.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plt.suptitle("Average Count-wise Accuracy Across 5 Runs", y=1.02)
#     plt.savefig("combined_accuracy_plots.png", dpi=150, bbox_inches='tight')
#     plt.show()

# if __name__ == "__main__":
#     main()