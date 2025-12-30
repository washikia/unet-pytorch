"""Plot training history saved by UNetTrainer.save_history().

Generates:
 - loss_plot.png (train vs val loss)
 - auc_plot.png (validation AUC vs epochs)
 - combined_plot.png with both graphs stacked
"""
import json
import os
import matplotlib.pyplot as plt


def load_history(history_path: str):
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_history(history: dict, output_dir: str = '.', show: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    val_auc = history.get('val_auc', [])

    epochs = list(range(1, len(train_loss) + 1))

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train loss')
    if len(val_loss) > 0:
        plt.plot(epochs[:len(val_loss)], val_loss, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    # AUC plot (validation)
    if len(val_auc) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(list(range(1, len(val_auc) + 1)), val_auc, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Validation ROC AUC')
        plt.title('Validation ROC AUC per Epoch')
        plt.grid(True)
        auc_path = os.path.join(output_dir, 'auc_plot.png')
        plt.savefig(auc_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # Combined plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    axes[0].plot(epochs, train_loss, label='Train loss')
    if len(val_loss) > 0:
        axes[0].plot(epochs[:len(val_loss)], val_loss, label='Val loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    if len(val_auc) > 0:
        axes[1].plot(list(range(1, len(val_auc) + 1)), val_auc, marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation ROC AUC')
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'No validation AUC recorded', ha='center')
        axes[1].set_axis_off()

    combined_path = os.path.join(output_dir, 'combined_plot.png')
    plt.tight_layout()
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    print(f"Saved plots to {output_dir}: {os.listdir(output_dir)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot training history saved by UNetTrainer')
    parser.add_argument('--history', type=str, default='checkpoints/training_history.json', help='Path to history JSON file')
    parser.add_argument('--output', type=str, default='predictions/plots', help='Directory to save plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')

    args = parser.parse_args()

    hist = load_history(args.history)
    plot_history(hist, output_dir=args.output, show=args.show)
