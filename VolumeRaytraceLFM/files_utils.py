from datetime import datetime
import os

def create_unique_directory(base_output_dir):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string in the desired format, e.g., 'YYYY-MM-DD_HH-MM-SS'
    dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    unique_output_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(unique_output_dir, exist_ok=True)
    print(f"Created the unique output directory {unique_output_dir}")
    return unique_output_dir
