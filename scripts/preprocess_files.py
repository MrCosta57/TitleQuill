import os
import glob

import hydra
from omegaconf import DictConfig


def fix_filenames(directory: str, old_ext: str, new_ext: str):

    print(f"Old extension: {old_ext}")
    print(f"New extension: {new_ext}")
    print(f"Search directory: {directory}")

    files = glob.glob(os.path.join(directory, f"*.{old_ext}"))
    files.sort()
    print(f"Found {len(files)} new files")
    pre_files = glob.glob(os.path.join(directory, f"*.{new_ext}"))
    print(f"Found {len(pre_files)} old files")

    # Iterate over each file
    for i, old_filename in enumerate(files, start=len(pre_files)):
        # Generate a new name and extension
        new_base_name = f"part_{i}.{new_ext}"
        # Construct the full new file path
        new_filename = os.path.join(directory, new_base_name)

        # Rename the file
        os.rename(old_filename, new_filename)
        print(f"Renamed: {old_filename} -> {new_filename}")

    print("Done!")


@hydra.main(version_base="1.3", config_path="../configs", config_name="scripts")
def main(cfg: DictConfig):
    # Small test
    directory = cfg.data.data_dir
    old_ext = cfg.old_ext
    new_ext = cfg.new_ext
    fix_filenames(directory, old_ext, new_ext)


if __name__ == "__main__":
    main()
