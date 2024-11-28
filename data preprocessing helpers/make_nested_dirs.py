import os

def create_nested_directories(base_dir, item_list, num):
    """
    Creates a new directory inside `base_dir` for each item in `item_list`.
    Inside each of these directories, creates 27 subdirectories named 1-27.
    
    Args:
        base_dir (str): The path to the base directory.
        item_list (list): List of names for the directories to be created.
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True,)
    
    for item in item_list:
        # Create the main directory for the current item
        main_dir = os.path.join(base_dir, item)
        os.makedirs(main_dir, exist_ok=True)
        
        # Create subdirectories
        for i in range(1, num + 1):
            sub_dir = os.path.join(main_dir, str(i))
            os.makedirs(sub_dir, exist_ok=True)
        
        print(f"Created directories for: {item}")

# Example usage
base_directory = "new_data"  # Replace with the desired base directory path
names = ['Apple', 'Intel', 'Lockheed Martin', 'Microsoft', 'Nvidia', 'Oracle', 'Shopify', 'Sony', 'Tesla', 'Uber']  # Replace with your list of items
num = 27 #replace with the number of subfolders the directory needs.
create_nested_directories(base_directory, names, num)