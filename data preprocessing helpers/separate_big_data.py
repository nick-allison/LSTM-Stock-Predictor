import pandas as pd
import os

def process_large_csv(input_file, output_dir, chunk_size=100000):
    """
    Reads a large CSV file in chunks and writes each chunk to a separate file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the chunked CSV files.
        chunk_size (int): Number of lines per chunk (default 100,000).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_number = 1
    
    try:
        # Read the CSV file in chunks
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            # Construct the output file name
            output_file = os.path.join(output_dir, f'chunk_{chunk_number}.csv')
            
            # Write the chunk to a CSV file
            chunk.to_csv(output_file, index=False)
            print(f"Written chunk {chunk_number} to {output_file}")
            
            chunk_number += 1
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("Processing completed successfully.")

# Example usage
input_file = ""  # Add the path to your large CSV file
output_dir = ""  # Add the desired output directory
process_large_csv(input_file, output_dir)