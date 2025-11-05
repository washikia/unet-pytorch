"""
Module to visualize predictions and create heatmaps.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from typing import Optional

from unet_pytorch.predict import UNetPredictor
from unet_pytorch.utils import im_to_tensor


def make_heatmap(coords, shape, sigma=3):
    """Create a heatmap from coordinates.
    
    Args:
        coords: List of (x, y) coordinate tuples.
        shape: Tuple of (height, width) for the heatmap.
        sigma: Standard deviation for Gaussian kernel.
    
    Returns:
        heatmap: 2D numpy array with values in [0, 1].
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    
    for (x, y) in coords:
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        heatmap += np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap


def extract_coordinates_from_prediction(pred_map: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Extract coordinates from prediction map where values exceed threshold.
    
    Args:
        pred_map: 2D numpy array with prediction values (0-1).
        threshold: Threshold value to consider as positive prediction.
    
    Returns:
        coords: List of (x, y) coordinate tuples.
    """
    # Find coordinates where prediction exceeds threshold
    y_coords, x_coords = np.where(pred_map >= threshold)
    coords = list(zip(x_coords, y_coords))
    return coords


def visualize_prediction(
    image_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.5,
    sigma: float = 3,
    show_plot: bool = True
):
    """Visualize model prediction with heatmap overlay.
    
    Args:
        image_path: Path to input image.
        model_path: Path to trained model checkpoint.
        output_path: Optional path to save the visualization.
        threshold: Threshold for extracting coordinates from predictions.
        sigma: Standard deviation for heatmap Gaussian kernel.
        show_plot: Whether to display the plot.
    """
    # Load model and make prediction
    predictor = UNetPredictor(model_path)
    pred_map = predictor.predict(image_path)
    
    # Load original image
    img = io.imread(image_path)
    if len(img.shape) == 3:
        # Convert RGB to grayscale for display
        from skimage import color
        img = color.rgb2gray(img)
    
    # Extract coordinates from prediction
    coords = extract_coordinates_from_prediction(pred_map, threshold=threshold)
    
    # Create heatmap from coordinates
    heatmap = make_heatmap(coords, pred_map.shape, sigma=sigma)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Prediction map (raw model output)
    im1 = axes[0, 1].imshow(pred_map, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Model Prediction (Raw)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Heatmap from coordinates
    im2 = axes[1, 0].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Heatmap (σ={sigma}, {len(coords)} points)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Overlay heatmap on original image
    axes[1, 1].imshow(img, cmap='gray', alpha=0.7)
    im3 = axes[1, 1].imshow(heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title('Overlay: Image + Heatmap')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_batch(
    image_dir: str,
    model_path: str,
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    sigma: float = 3,
    extension: str = 'tif'
):
    """Visualize predictions for all images in a directory.
    
    Args:
        image_dir: Directory containing input images.
        model_path: Path to trained model checkpoint.
        output_dir: Optional directory to save visualizations.
        threshold: Threshold for extracting coordinates from predictions.
        sigma: Standard deviation for heatmap Gaussian kernel.
        extension: File extension to look for (default: 'tif').
    """
    import glob
    
    # Get all image files
    image_files = glob.glob(os.path.join(image_dir, f'*.{extension}'))
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {image_dir} with extension .{extension}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        output_path = None
        if output_dir:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
        
        visualize_prediction(
            image_path=image_path,
            model_path=model_path,
            output_path=output_path,
            threshold=threshold,
            sigma=sigma,
            show_plot=False
        )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize U-Net predictions with heatmaps')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output path for single image or directory for batch')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for coordinate extraction')
    parser.add_argument('--sigma', type=float, default=3, help='Sigma for heatmap Gaussian kernel')
    parser.add_argument('--extension', type=str, default='tif', help='File extension for batch processing')
    
    args = parser.parse_args()
    
    if args.image:
        visualize_prediction(
            image_path=args.image,
            model_path=args.model,
            output_path=args.output,
            threshold=args.threshold,
            sigma=args.sigma
        )
    elif args.image_dir:
        visualize_batch(
            image_dir=args.image_dir,
            model_path=args.model,
            output_dir=args.output,
            threshold=args.threshold,
            sigma=args.sigma,
            extension=args.extension
        )
    else:
        print("Please provide either --image or --image_dir")
        parser.print_help()

