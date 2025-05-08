import os
import json
import hydra
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.utils.majority_vote import majority_vote
from src.utils.metrics import save_metrics
from src.classification import initialize_data

def output_statistics(data_loaders, split_names, output_dir): 
    labels = ["Normal", "Depression", "Suicidal", "Anxiety", "Stress", "Bipolar", "Personality disorder"]

    counts = {}
    for split_name in split_names:
        counts[split_name] = {}
        for label in labels:
            counts[split_name][label] = 0

    for data_loader, split_name in zip(data_loaders, split_names):
        for _, label in data_loader:
            counts[split_name][label[0]] += 1
    
    cleaned_counts = {}
    for splits, label_count_dict in counts.items():
        cleaned_counts[splits] = list(label_count_dict.values())

    # Dump train, test, val split
    with open(os.path.join(output_dir, "split_count.txt"), "w") as f:
        total_sum = 0
        for _, v in cleaned_counts.items():
            total_sum += sum(v)

        for k, v in cleaned_counts.items():
            f.write(f"{k} set: {sum(v)}, {100*sum(v)/total_sum}%\n")
            
        f.write(f"total: {total_sum}")

    # Dump examples per class
    with open(os.path.join(output_dir, "class_count.txt"), "w") as f:
        total_sum = 0
        for _, v in cleaned_counts.items():
            total_sum += sum(v)

        for k, v in cleaned_counts.items():
            f.write(f"{k} set:\n")
            c = 0
            for label_count in v:
                f.write(f"\t{labels[c]}: {label_count}\n")
                c += 1

        f.write(f"Total: \n")
        all_counts_matrix = []
        for _, v in cleaned_counts.items():
            all_counts_matrix.append(v)
        all_counts_matrix = np.array(all_counts_matrix)
        summed_counts = np.sum(all_counts_matrix, axis=0)
        for i, label in enumerate(labels):
            f.write(f"\t{label}: {summed_counts[i]}\n")

        f.write(f"Total percentages: \n")
        for i, label in enumerate(labels):
            f.write(f"\t{label}: {100*summed_counts[i]/total_sum}%\n")

    # **BEAUTIFUL FUCKING PLOT**
    labels_for_plot = ["Normal", "Depression", "Suicidal", "Anxiety", "Stress", "Bipolar", "P. D."]
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.3, 2.5))  
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    width = 0.25
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    offsets = [-width, 0, width]

    for i, split in enumerate(split_names):
        ax.bar(
            [xi + offsets[i] for xi in x],
            cleaned_counts[split],
            width=width,
            alpha=0.4,
            label=split,
            color=colors[i],
            edgecolor='black',
            linewidth=1.0
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_for_plot, rotation=20, ha='right')
    ax.set_ylabel("Number of Samples")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300, bbox_inches='tight')

@hydra.main(version_base=None, config_path="../config", config_name="data_eval.yaml")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir

    parser_obj = initialize_data(cfg.data)
    train_loader, test_loader, val_loader = parser_obj.fill_dataloaders()

    output_statistics(data_loaders=[train_loader, test_loader, val_loader],
                      split_names=["Train", "Test", "Validation"],
                      output_dir=output_dir)

if __name__ == "__main__":
    main()