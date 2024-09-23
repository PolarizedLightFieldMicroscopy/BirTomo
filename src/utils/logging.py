import sys

# Function to redirect print statements to a log file
def redirect_output_to_log(log_file_path):
    log_file = open(log_file_path, "w")
    sys.stdout = log_file
    # sys.stderr = log_file
    return log_file

# Restore the standard output to its original state
def restore_output(log_file):
    sys.stdout = sys.__stdout__
    log_file.close()
