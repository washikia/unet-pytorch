"""
Example script to visualize predictions with heatmaps.
"""
from unet_pytorch.visualize import visualize_prediction, visualize_batch

# Example 1: Visualize a single image
# visualize_prediction(
#     image_path="data/inputs/image_0.tif",
#     model_path="checkpoints/best_model.pth",
#     output_path="predictions/image_0_heatmap.png",
#     threshold=0.5,
#     sigma=3,
#     show_plot=True
# )

# Example 2: Process all images in a directory
# visualize_batch(
#     image_dir="data/inputs",
#     model_path="checkpoints/best_model.pth",
#     output_dir="predictions",
#     threshold=0.5,
#     sigma=3,
#     extension="tif"
# )

if __name__ == "__main__":
    # Update these paths to match your setup
    model_path = "D:\\washik_personal\\projects\\Unet\\unet-pytorch\\checkpoints\\best_model.pth"
    image_dir = "D:\\washik_personal\\projects\\Unet\\unet-pytorch\\data\\inputs"
    output_dir = "D:\\washik_personal\\projects\\Unet\\unet-pytorch\\predictions"
    
    # Process all images
    visualize_batch(
        image_dir=image_dir,
        model_path=model_path,
        output_dir=output_dir,
        threshold=0.5,
        sigma=3,
        extension="tif"
    )

