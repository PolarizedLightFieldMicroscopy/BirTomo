"""Module for redirecting print statements to a log file."""

import sys


def redirect_output_to_log(log_file_path):
    """Append print statements to a log file."""
    log_file = open(log_file_path, "a")
    sys.stdout = log_file
    return log_file


def restore_output(log_file):
    """Restore the standard output to its original state."""
    sys.stdout = sys.__stdout__
    log_file.close()
