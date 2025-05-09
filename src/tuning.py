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

from src.classification import initialize_data, initialize_naive_bayes, initialize_softmax
from src.utils.metrics import save_metrics

@hydra.main(version_base=None, config_path="../config", config_name="tuning")
def main(cfg: DictConfig):
    output_dir = cfg.results_dir

    # Initialize data
    parser_obj = initialize_data(cfg.data)
    train_loader, test_loader, val_loader = parser_obj.fill_dataloaders()

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

    elif "softmax" in cfg.classifier._target_:
        text_vocab, label_vocab = parser_obj.build_vocab(train_loader, test_loader, val_loader)

        train_loader = parser_obj.vectorize_dataloaders(train_loader)
        test_loader = parser_obj.vectorize_dataloaders(test_loader)
        val_loader = parser_obj.vectorize_dataloaders(val_loader)

        classifier = initialize_softmax(cfg.classifier, text_vocab, label_vocab)

        # tuning these parameters:
        latent_dims = [8, 16, 32, 64]
        embedding_dims = [50, 100, 128, 300]

        accuracy_grid = np.zeros((len(latent_dims), len(embedding_dims)))
        precision_grid = np.zeros_like(accuracy_grid)
        recall_grid = np.zeros_like(accuracy_grid)
        f1_grid = np.zeros_like(accuracy_grid)

        for i, latent_dim in enumerate(latent_dims):
            for j, embedding_dim in enumerate(embedding_dims):
                l_e_output_dir = os.path.join(output_dir, f"ld_{latent_dim}_ed_{embedding_dim}")
                os.makedirs(l_e_output_dir, exist_ok=True)

                print(f"Running ld: {latent_dim}, ed: {embedding_dim}")

                classifier.config.latent_dim = latent_dim
                classifier.config.embedding_dim = embedding_dim

                epoch_losses = classifier.train(train_loader)

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

                plt.plot(np.arange(len(epoch_losses)), epoch_losses, color=colors[0])
                plt.xlabel('Epochs')
                plt.ylabel('Cross Entropy Loss')
                plt.title('Learning Curve')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(l_e_output_dir, "training_loss.png"))

                possible_labels = []
                for _, token in enumerate(label_vocab.get_itos()):
                    possible_labels.append(token)

                test_pred = classifier.predict(val_loader)

                pred_labels = []
                for p in test_pred:
                    pred_labels.append(possible_labels[p])

                _, true_labels = parser_obj.tokens_to_words(val_loader)

                a, p, r, f = save_metrics(true_labels, pred_labels, l_e_output_dir)

                accuracy_grid[i, j] = a
                precision_grid[i, j] = p
                recall_grid[i, j] = r
                f1_grid[i, j] = f

        plot_3d_surface(embedding_dims, latent_dims, accuracy_grid, "Accuracy Surface", "Accuracy", output_dir)
        plot_3d_surface(embedding_dims, latent_dims, precision_grid, "Precision Surface", "Precision", output_dir)
        plot_3d_surface(embedding_dims, latent_dims, recall_grid, "Recall Surface", "Recall", output_dir)
        plot_3d_surface(embedding_dims, latent_dims, f1_grid, "F1 Score Surface", "F1 Score", output_dir)


def plot_3d_surface(X, Y, Z, title, zlabel, output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X_grid, Y_grid = np.meshgrid(X, Y)

    ax.plot_surface(X_grid, Y_grid, Z, cmap='coolwarm', edgecolor='k', linewidth=0.5, alpha=0.7)
    ax.scatter(X_grid, Y_grid, Z, color='yellow', s=20)

    contour = ax.contourf(X_grid, Y_grid, Z, zdir='z', offset=np.min(Z) - 0.1, cmap='coolwarm', alpha=0.8)

    ax.set_xlabel('Embedding Dim')
    ax.set_ylabel('Latent Dim')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
 
    ax.set_zlim(np.min(Z) - 0.1, np.max(Z) + 0.05)

    cbar = fig.colorbar(contour, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label(zlabel)

    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{title}.png"))

if __name__ == "__main__":
    main()