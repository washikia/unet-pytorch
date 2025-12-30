from pathlib import Path

INPUT_DIR = Path("D:\\washik_personal\\projects\\Unet\\unet-pytorch\\data\\inputs")
TARGET_DIR = Path("D:\\washik_personal\\projects\\Unet\\unet-pytorch\\data\\targets")

INPUT_EXT = ".tif"
TARGET_EXT = ".png"

def main(delete: bool = True):
    input_files = {p.stem: p for p in INPUT_DIR.glob(f"*{INPUT_EXT}")}
    target_files = {p.stem: p for p in TARGET_DIR.glob(f"*{TARGET_EXT}")}

    input_stems = set(input_files.keys())
    target_stems = set(target_files.keys())

    # Files without a pair
    inputs_without_target = input_stems - target_stems
    targets_without_input = target_stems - input_stems

    print(f"Inputs without targets: {len(inputs_without_target)}")
    print(f"Targets without inputs: {len(targets_without_input)}")

    for stem in inputs_without_target:
        path = input_files[stem]
        print(f"Removing input: {path}")
        if delete:
            path.unlink()

    for stem in targets_without_input:
        path = target_files[stem]
        print(f"Removing target: {path}")
        if delete:
            path.unlink()

    # Final check
    final_inputs = len(list(INPUT_DIR.glob(f"*{INPUT_EXT}")))
    final_targets = len(list(TARGET_DIR.glob(f"*{TARGET_EXT}")))

    print("\nFinal counts:")
    print(f"Inputs : {final_inputs}")
    print(f"Targets: {final_targets}")

    assert final_inputs == final_targets, "Mismatch after cleanup!"

    print("\n✅ Cleanup complete. Folders are synchronized.")

if __name__ == "__main__":
    main(delete=True)
