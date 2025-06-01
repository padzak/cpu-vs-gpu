import os
import shutil
from pathlib import Path
import argparse

def duplicate_png_files(root_dir: str, N: int):
    root_path = Path(root_dir)
    new_root_path = root_path.parent / f"{root_path.name}_x{N + 1}"
    shutil.copytree(root_path, new_root_path, dirs_exist_ok=True)

    for class_dir in new_root_path.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            existing_files = sorted(class_dir.glob(f"{label}*.png"))

            # Determine the highest index number to avoid overwriting
            max_index = 0
            for file in existing_files:
                try:
                    index = int(file.stem.split('.')[-1])
                    max_index = max(max_index, index)
                except ValueError:
                    continue

            for file in existing_files:
                for i in range(N):
                    max_index += 1
                    new_filename = f"{label}.{max_index:04d}.png"
                    new_file_path = class_dir / new_filename
                    shutil.copy(file, new_file_path)
                    print(f"Copied {file.name} -> {new_filename} in {class_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duplicate PNG files in dataset directories.")
    parser.add_argument("dataset_root", type=str, help="Path to the root training directory (e.g., 'train')")
    parser.add_argument("duplication_factor", type=int, help="Number of times to duplicate each file")

    args = parser.parse_args()
    duplicate_png_files(args.dataset_root, args.duplication_factor)
