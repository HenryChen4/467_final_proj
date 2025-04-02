import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_confusion_matrix(confusion_matrix, 
                          display_labels,
                          output_dir):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    
    plt.rcParams['font.family'] = 'Times New Roman' 
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Confusion Matrix", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir)

def save_metrics(y_true, y_pred, results_dir):
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro')}")
    print(f"Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0)}")
    print(f"F1: {f1_score(y_true, y_pred, average='macro')}")

    with open(os.path.join(results_dir, "scores.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred, average='macro')}\n")
        f.write(f"Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0)}\n")
        f.write(f"F1: {f1_score(y_true, y_pred, average='macro')}\n")

    cm = confusion_matrix(y_true, y_pred)
    
    display_labels = sorted(set(y_true) | set(y_pred))
    show_confusion_matrix(confusion_matrix=cm,
                          display_labels=display_labels,
                          output_dir=os.path.join(results_dir, "confusion_matrix.png"))