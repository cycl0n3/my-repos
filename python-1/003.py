import os
import filecmp

def compare_folders(folder1, folder2):
    """
    Compares two folders, returning lists of matching, mismatched, and missing files.

    This function takes in two folder paths as arguments and returns a tuple containing three lists:
    - match: List of filenames that exist in both folders and have identical content.
    - mismatch: List of filenames that exist in both folders but have different content.
    - missing: List of filenames that exist in one folder but not the other.

    Args:
        folder1 (str): Path to the first folder.
        folder2 (str): Path to the second folder.

    Returns:
        tuple: A tuple containing three lists:
            - match: List of filenames that exist in both folders and have identical content.
            - mismatch: List of filenames that exist in both folders but have different content.
            - missing: List of filenames that exist in one folder but not the other.
    """
    match, mismatch, errors = filecmp.cmpfiles(folder1, folder2, [], shallow=False)

    missing_in_folder1 = [os.path.join(folder2, filename) for filename in os.listdir(folder2)
                          if filename not in errors and filename not in match]
    missing_in_folder2 = [os.path.join(folder1, filename) for filename in os.listdir(folder1)
                          if filename not in errors and filename not in match]

    return match, mismatch + errors, missing_in_folder1, missing_in_folder2

def display_diff(file1, file2):
    """
    Displays the difference between two files using the difflib module.

    This function takes in two file paths as arguments and prints the difference between them using the difflib module.

    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.
    """
    import difflib

    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        data1 = f1.readlines()
        data2 = f2.readlines()

    diff = difflib.unified_diff(data1, data2, fromfile=file1, tofile=file2)
    print("\n".join(diff))

if __name__ == "__main__":
    folder1 = "C:\\Users\\Millind\\Videos\\100CANON-0"
    folder2 = "C:\\Users\\Millind\\Videos\\100CANON"

    match, mismatch, missing_in_folder1, missing_in_folder2 = compare_folders(folder1, folder2)

    print("\nMatching files:")
    for filename in match:
        print(filename)

    print("\nMismatched/error files:")
    for filename in mismatch:
        print(filename)
        display_diff(os.path.join(folder1, filename), os.path.join(folder2, filename))

    print("\nMissing files in folder 1:")
    for filename in missing_in_folder1:
        print(filename)

    print("\nMissing files in folder 2:")
    for filename in missing_in_folder2:
        print(filename)


def add(x, y):
    return x + y