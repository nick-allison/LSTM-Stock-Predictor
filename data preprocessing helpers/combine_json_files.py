import os
import json
import glob
import shutil

def combine_json_files(root_folder):
    # Get list of subfolders in the root folder
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        print(f"Processing subfolder: {subfolder_name}")
        # Use the existing load_sentiment_data function logic to read JSON files
        sentiment_data = []
        urls_seen = set()
        json_files = glob.glob(os.path.join(subfolder, "*.json"))
        
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                try:
                    # Attempt to load the entire content as JSON
                    data_loaded = json.loads(file_content)

                    if isinstance(data_loaded, list):
                        # The file contains a JSON array
                        for data in data_loaded:
                            if 'url' in data and data['url'] not in urls_seen:
                                urls_seen.add(data['url'])
                                sentiment_data.append(data)
                    elif isinstance(data_loaded, dict):
                        # The file contains a single JSON object
                        data = data_loaded
                        if 'url' in data and data['url'] not in urls_seen:
                            urls_seen.add(data['url'])
                            sentiment_data.append(data)
                    else:
                        print(f"Unsupported JSON format in file {file}.")
                except json.JSONDecodeError:
                    # Handle concatenated JSON objects
                    try:
                        idx = 0
                        length = len(file_content)
                        decoder = json.JSONDecoder()
                        while idx < length:
                            data, idx = decoder.raw_decode(file_content, idx)
                            if 'url' in data and data['url'] not in urls_seen:
                                urls_seen.add(data['url'])
                                sentiment_data.append(data)
                            # Skip any whitespace between JSON objects
                            while idx < length and file_content[idx].isspace():
                                idx += 1
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file}: {e}")
                        continue  # Skip malformed JSON files
        
        # Write combined data to a new JSON file in the root directory
        output_file = os.path.join(root_folder, f"{subfolder_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(sentiment_data, f_out, ensure_ascii=False)
        print(f"Combined data written to {output_file}")
        
        # Remove the subfolder
        shutil.rmtree(subfolder)
        print(f"Removed subfolder {subfolder_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Combine JSON files from subfolders.')
    parser.add_argument('root_folder', type=str, help='Path to the root folder')

    args = parser.parse_args()

    combine_json_files(args.root_folder)