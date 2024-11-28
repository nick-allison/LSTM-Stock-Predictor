import os
import shutil

def gather_files_into_one_folder(source_dir, destination_dir):
    """
    Moves all files from all subdirectories of `source_dir` into `destination_dir`, renaming files to avoid duplicates.
    
    Args:
        source_dir (str): The directory to traverse and gather files from.
        destination_dir (str): The directory to collect all files into.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    file_counter = 1  # To keep track of new filenames
    for root, _, files in os.walk(source_dir):
        for file in files:
            old_file_path = os.path.join(root, file)
            new_file_name = f"file{file_counter}{os.path.splitext(file)[1]}"  # Preserve original file extension
            new_file_path = os.path.join(destination_dir, new_file_name)

            shutil.copy2(old_file_path, new_file_path)  # Copy file to the new folder
            print(f"Moved: {old_file_path} -> {new_file_path}")

            file_counter += 1

if __name__ == "__main__":
    source_directory = ""  #Enter source directory here
    destination_directory = "" #Enter destination directory here
    
    gather_files_into_one_folder(source_directory, destination_directory)