import subprocess
import re
import os
import json
import hydra

import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from src.classification import initialize_data, initialize_naive_bayes
from src.utils.metrics import save_metrics

@hydra.main(version_base=None, config_path="../config", config_name="tuning")
def main(cfg: DictConfig):
    output_dir = cfg.results_dir

    # Initialize data
    parser_obj = initialize_data(cfg.data)
    train_loader, _, val_loader = parser_obj.fill_dataloaders()

    # Initialize and tune classifiers
    classifier = None
    if "naive_bayes" in cfg.classifier._target_:
        classifier = initialize_naive_bayes(cfg.classifier)

        # Train and gather validation result        
        alphas = [0.0, 1e-5, 1e-3, 0.01, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]

        accuracies = []
        precisions = []
        recall = []
        f1 = []

        for alpha in tqdm(alphas):
            alpha_output_dir = os.path.join(output_dir, f"{alpha}")
            os.makedirs(alpha_output_dir, exist_ok=True)

            print(f"Running alpha={alpha}")

            classifier.alpha = alpha
            classifier.train(train_loader)

            text_val_labels = [label[0] for _, label in val_loader]
            val_labels = classifier.label_encoder.transform(text_val_labels)
            val_pred = classifier.predict(val_loader)

            val_labels = classifier.label_encoder.inverse_transform(val_labels)
            val_pred = classifier.label_encoder.inverse_transform(val_pred)

            a, p, r, f = save_metrics(val_labels, val_pred, alpha_output_dir)

            accuracies.append(a)
            precisions.append(p)
            recall.append(r)
            f1.append(f)

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
        colors = ['tab:blue', 'tab:orange', 'tab:green']
 
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(alphas, accuracies, marker='o', label='Accuracy', color=colors[0])
        plt.plot(alphas, precisions, marker='s', label='Precision', color=colors[1])
        plt.plot(alphas, recall, marker='^', label='Recall', color=colors[2])
        plt.plot(alphas, f1, marker='D', label='F1 Score', color='black') 

        # plt.xscale('log')
        plt.xlabel('Laplace Smoothing (α)')
        plt.ylabel('Score')
        plt.title('NB Performance vs. Smoothing')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "tuning_results.png"))

if __name__ == "__main__":
    main()