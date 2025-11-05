"""
Module to make predictions using the U-Net model.
"""
import numpy as np
import torch

from unet_pytorch.model import UNet
from unet_pytorch.utils import im_to_tensor


class UNetPredictor:
    """Predictor class for the U-Net model."""

    def __init__(self, model_file: str) -> None:
        """Initialise the U-Net predictor.

        Args:
            path: Path to the model.
        """
        self.model_file = model_file

        if torch.backends.mps.is_available():
            self.device = torch.device('mps') # Apple silicon
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = UNet().to(self.device)
        checkpoint = torch.load(self.model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_weights'])
        # Ensure model is in float32 mode
        self.model = self.model.float()
        self.model.eval()

    def predict(self, image_file: str) -> np.ndarray:
        """Make a prediction using the U-Net.
        Args:
            image_file: Path to the image.
        Returns:
            pred: Prediction as numpy array (single channel, values 0-1).
        """
        im = im_to_tensor(image_file)
        # Ensure float32 and add batch dimension
        im = im.unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            pred = self.model(im)

        # For single channel output, squeeze and return as numpy
        pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
        return pred
    
    def predict_probability(self, image_file: str) -> np.ndarray:
        """Get probability/confidence map from the model.
        Args:
            image_file: Path to the image.
        Returns:
            prob_map: Probability map as numpy array (values 0-1).
        """
        return self.predict(image_file)

    def predict_batch(self, image_files: list[str]) -> list[np.ndarray]:
        """Make predictions for a batch of images.
        
        Args:
            image_files: List of paths to the images.
        Returns:
            preds: List of predictions for each image.
        """
        preds = []
        for image_file in image_files:
            pred = self.predict(image_file)
            preds.append(pred)

        return preds
