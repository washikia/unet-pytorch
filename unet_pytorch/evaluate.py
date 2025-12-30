"""
CLI to evaluate a trained model on the test split and save predictions.
"""
import os
import argparse

from unet_pytorch.train import UNetTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a U-Net model on the test set and save predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (e.g., checkpoints/best_model.pth)")
    parser.add_argument("--inputs", type=str, default="data/inputs", help="Path to input images folder")
    parser.add_argument("--targets", type=str, default="data/targets", help="Path to target masks folder")
    parser.add_argument("--output", type=str, default="predictions/test", help="Directory to save predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to binarize prediction for IoU/mask saving")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction to use for validation split when constructing dataset")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction to use for test split when constructing dataset")

    args = parser.parse_args()

    # Ensure output dir exists
    os.makedirs(args.output, exist_ok=True)

    model_dir = os.path.dirname(args.model) or "."

    trainer = UNetTrainer(
        inputs_path=args.inputs,
        targets_path=args.targets,
        model_path=model_dir,
        batch_size=1,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    # Load the checkpoint (accepts absolute path too)
    trainer.load_checkpoint(args.model)

    # Run evaluation and save outputs
    trainer.evaluate_test(threshold=args.threshold, output_dir=args.output)


if __name__ == "__main__":
    main()
