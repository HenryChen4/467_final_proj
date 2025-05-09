import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_confusion_matrix(confusion_matrix, 
                          display_labels,
                          output_dir):
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

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    disp.plot(
        cmap="Blues",  
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.2)
    plt.savefig(output_dir, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(y_true, y_pred, results_dir):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0)}")
    print(f"Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0)}")
    print(f"F1: {f1_score(y_true, y_pred, average='macro', zero_division=0)}")

    with open(os.path.join(results_dir, "scores.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0)}\n")
        f.write(f"Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0)}\n")
        f.write(f"F1: {f1_score(y_true, y_pred, average='macro', zero_division=0)}\n")

    cm = confusion_matrix(y_true, y_pred)
    
    display_labels = sorted(set(y_true) | set(y_pred))
    show_confusion_matrix(confusion_matrix=cm,
                          display_labels=display_labels,
                          output_dir=os.path.join(results_dir, "confusion_matrix.png"))
    
    return accuracy, precision, recall, f1