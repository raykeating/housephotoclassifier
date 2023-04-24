import os
import shutil

source_dir = "C:\\Users\\rayke\\Downloads\\archive(1)\\raw-img"
dest_dir = "C:\\Users\\rayke\\Downloads\\archive(1)\\raw-img-small"

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop over subdirectories in source directory
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # Loop over files in subdirectory and copy 10 files to destination
    file_counter = 0
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)
        if os.path.isfile(file_path):
            # Copy file to destination
            dest_file_path = os.path.join(dest_dir, file)
            shutil.copy2(file_path, dest_file_path)
            file_counter += 1
            if file_counter == 10:
                break
#

