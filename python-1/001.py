import os
import shutil
import fnmatch

def list_files_with_ru(directory):
    files_with_ru = []
    files_without_ru = []

    for root, _, files in os.walk(directory):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            if "(ru)" in file_name:
                files_with_ru.append(full_path)
            else:
                files_without_ru.append(full_path)

    return files_with_ru, files_without_ru

def move_files_with_ru(files_with_ru, target_dir_ru):
    for file_path in files_with_ru:
        relative_path = os.path.relpath(file_path, start=directory)
        target_path = os.path.join(target_dir_ru, relative_path)
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        print("copying: ", target_path)
        
        try:
          shutil.copy2(file_path, target_path)
        except:
            print('error copying: ', target_path)

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    files_with_ru, files_without_ru = list_files_with_ru(directory)

    # print("\nFiles containing '(ru)':")
    # for file_path in files_with_ru:
    #     print(file_path)

    # print("\nFiles not containing '(ru)':")
    # for file_path in files_without_ru:
    #     print(file_path)

    target_dir_en = os.path.join(directory, "EN")
    target_dir_ru = os.path.join(directory, "RU")
    
    os.makedirs(target_dir_en, exist_ok=True)
    os.makedirs(target_dir_ru, exist_ok=True)

    move_files_with_ru(files_with_ru, target_dir_ru)
    move_files_with_ru(files_without_ru, target_dir_en)

    print("\nFiles moved successfully!")
