"""
Module to train the U-Net model.
"""
import os
import dataclasses

import torch
from torch import nn

from unet_pytorch.model import UNet
from unet_pytorch.data import UNetDataset, UNetDataLoader


@dataclasses.dataclass(init=False)
class UNetTrainingConfig:
    """Configuration class for U-Net training components."""

    def __init__(self, model: UNet, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Initialise the U-Net training configuration."""
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_fn = nn.MSELoss() # changed from CrossEntropyLoss to MSELoss
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode='min', patience=5, factor=0.5
        )


class UNetTrainer:
    """Trainer class for the U-Net model."""

    def __init__(
            self,
            inputs_path: str,
            targets_path: str,
            model_path: str,
            batch_size: int = 1
        ) -> None:
        """Initialise the U-Net trainer.

        Args:
            inputs_path: Path to the input images.
            targets_path: Path to the target images.
            model_path: Path to save the model checkpoints.
            batch_size: Batch size for training.
        """
        self.paths = {
            'inputs': inputs_path,
            'targets': targets_path,
            'model': model_path,
        }
        self.batch_size = batch_size

        self.dataset = UNetDataset(self.paths['inputs'], self.paths['targets'])
        self.data_loader = UNetDataLoader(
            self.dataset, self.batch_size
        )

        if torch.backends.mps.is_available():
            self.device = torch.device('mps') # Apple silicon
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")

        self.model = UNet().to(self.device)

        self.training_config = UNetTrainingConfig(self.model)

        # Track metrics per epoch; val_auc stores validation ROC-AUC values
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    def evaluate_test(self, threshold: float = 0.5, output_dir: str | None = None) -> dict:
        """Run the model on the held-out test set, compute metrics and optionally save predictions.

        Args:
            threshold: threshold for binary masks when computing IoU and saving masks.
            output_dir: directory to save probability maps and masks (if provided).

        Returns:
            Dictionary with average metrics: {'avg_mse':..., 'avg_iou':..., 'n':...}
        """
        from skimage import io, img_as_ubyte
        import numpy as np
        from unet_pytorch.utils import im_to_tensor
        import os

        test_dataset = self.dataset.get_test_dataset
        if len(test_dataset) == 0:
            print("No test set available (test split may be empty).")
            return {'avg_mse': float('nan'), 'avg_iou': float('nan'), 'n': 0}

        self.model.eval()
        mse_total = 0.0
        iou_total = 0.0
        n = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for i, input_path in enumerate(test_dataset.x):
                # Forward
                inp = im_to_tensor(input_path).unsqueeze(0).float().to(self.device)
                pred = self.model(inp).squeeze(0).squeeze(0).detach().cpu().numpy()

                # Load target and normalise
                target = io.imread(test_dataset.y[i])
                if target.max() > 1.0:
                    target = target.astype(np.float32) / 255.0

                # Metrics
                mse = float(((pred - target) ** 2).mean())
                pred_bin = (pred > threshold).astype(np.float32)
                inter = (pred_bin * target).sum()
                union = pred_bin.sum() + target.sum() - inter
                iou = float(inter / union) if union > 0 else 0.0

                mse_total += mse
                iou_total += iou
                n += 1

                # Save outputs
                if output_dir:
                    base = os.path.splitext(os.path.basename(input_path))[0]
                    prob_path = os.path.join(output_dir, f"{base}_prob.png")
                    mask_path = os.path.join(output_dir, f"{base}_mask.png")
                    io.imsave(prob_path, img_as_ubyte(np.clip(pred, 0, 1)))
                    io.imsave(mask_path, img_as_ubyte(pred_bin))

        avg_mse = mse_total / n if n > 0 else float('nan')
        avg_iou = iou_total / n if n > 0 else float('nan')

        print(f"Test results — n={n}, Avg MSE={avg_mse:.6f}, Avg IoU={avg_iou:.6f}")

        return {'avg_mse': avg_mse, 'avg_iou': avg_iou, 'n': n}

    def _train_one_epoch(self) -> float:
        """Train the model for one epoch.
        
        Returns:
            Average loss for the epoch.
        """
        train_loader = self.data_loader.get_train_loader
        self.model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            # For MSE loss, both pred and y should be float with matching shapes
            # y shape: (batch, 1, H, W), pred shape: (batch, 1, H, W)
            loss = self.training_config.loss_fn(pred, y)

            self.training_config.optimiser.zero_grad()
            loss.backward()
            self.training_config.optimiser.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(train_loader.dataset)

    def _evaluate(self) -> tuple[float, float]:
        """Evaluate the model on the validation set and compute ROC-AUC.
        
        Returns:
            avg_loss: Average MSE loss for the validation set.
            auc: ROC-AUC computed on all pixels (or nan if not computable).
        """
        val_loader = self.data_loader.get_val_loader
        self.model.eval()
        total_loss = 0.0
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                # For MSE loss, both pred and y should be float with matching shapes
                loss = self.training_config.loss_fn(pred, y)

                total_loss += loss.item() * x.size(0)

                # Collect for AUC computation
                preds_list.append(pred.detach().cpu().numpy())
                targets_list.append(y.detach().cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)

        # Flatten arrays and compute ROC AUC across pixels
        try:
            import numpy as np
            from sklearn.metrics import roc_auc_score

            preds_arr = np.concatenate(preds_list, axis=0).ravel()
            targets_arr = np.concatenate(targets_list, axis=0).ravel()

            if targets_arr.size == 0 or preds_arr.size == 0 or len(np.unique(targets_arr)) < 2:
                auc = float('nan')
            else:
                auc = float(roc_auc_score(targets_arr, preds_arr))
        except Exception:
            auc = float('nan')

        return avg_loss, auc

    def train(self, epochs: int=50, save_interval: int=10) -> None:
        """Train the U-Net model.

        Args:
            epochs: Number of epochs to train for.
            save_interval: Interval for saving checkpoints.
        """
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self._train_one_epoch()
            self.history['train_loss'].append(train_loss)

            val_loss, val_auc = self._evaluate()
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)

            # Persist history every epoch
            self.save_history()

            # Step scheduler by validation loss
            self.training_config.scheduler.step(val_loss)

            print(
                f'Epoch {epoch + 1}/{epochs}, '
                f'Train loss: {train_loss:.4f}, '
                f'Validation loss: {val_loss:.4f}, '
                f'Val AUC: {val_auc if not (val_auc!=val_auc) else float("nan"):.4f}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, 'best_model.pth')

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, val_loss, f'checkpoint_epoch_{epoch + 1}.pth')

        self.save_checkpoint(epochs, val_loss, 'final_model.pth')
        print('Training complete!')

    def save_checkpoint(self, epoch: int, val_loss: float, filename: str) -> None:
        """Save the model checkpoint.
        
        Args:
            epoch: Current epoch number.
            val_loss: Validation loss for the current epoch.
            filename: Filename for the checkpoint.
        """
        torch.save({
            'epoch': epoch + 1,
            'model_weights': self.model.state_dict(),
            'optimiser_weights': self.training_config.optimiser.state_dict(),
            'loss': val_loss,
            'history': self.history,
        }, os.path.join(self.paths['model'], filename)
        )
        # Also persist the training history when saving checkpoints
        try:
            self.save_history()
        except Exception:
            pass

    def save_history(self, json_name: str = 'training_history.json', csv_name: str = 'training_history.csv') -> None:
        """Save training history to JSON and CSV in the model directory.
        Args:
            json_name: JSON filename.
            csv_name: CSV filename.
        """
        import json, csv

        model_dir = self.paths['model']
        os.makedirs(model_dir, exist_ok=True)

        # Save JSON
        json_path = os.path.join(model_dir, json_name)
        with open(json_path, 'w') as jf:
            json.dump(self.history, jf)

        # Save CSV
        csv_path = os.path.join(model_dir, csv_name)
        epochs = len(self.history.get('train_loss', []))
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_auc'])
            for i in range(epochs):
                t = self.history.get('train_loss', [None]*epochs)[i]
                v = self.history.get('val_loss', [None]*epochs)[i] if i < len(self.history.get('val_loss', [])) else ''
                a = self.history.get('val_auc', [None]*epochs)[i] if i < len(self.history.get('val_auc', [])) else ''
                writer.writerow([i + 1, t, v, a])

    def load_checkpoint(self, filename: str) -> tuple[int, float]:
        """Load the model checkpoint.
        
        Args:
            filename: Filename for the checkpoint.
        
        Returns:
            start_epoch: Starting epoch number.
            best_val_loss: Best validation loss.
        """
        checkpoint_path = os.path.join(self.paths['model'], filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            print(f'Checkpoint {checkpoint_path} not found.')
            return 0, float('inf')
        self.model.load_state_dict(checkpoint['model_weights'])
        self.training_config.optimiser.load_state_dict(checkpoint['optimiser_weights'])
        start_epoch = checkpoint['epoch'] - 1
        best_val_loss = checkpoint['loss']
        self.history = checkpoint['history']
        print(
            f'Loaded checkpoint from epoch {start_epoch + 1} '
            f'with validation loss {best_val_loss:.4f}'
        )
        return start_epoch, best_val_loss


# trainer = UNetTrainer(
#     inputs_path="path/to/inputs",  # folder with .tif files
#     targets_path="path/to/targets", # folder with .png files
#     model_path="path/to/save/models",
#     batch_size=4
# )
# trainer.train(epochs=50, save_interval=10)
