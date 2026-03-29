"""
rowデータ全てを検索に使用するのはコストが高いので、ランダムに問題を選んでコピーするスクリプト。
コピー先は dest_dir/run_20251012-1630/ のように、実行日時を含むサブディレクトリを作成してそこにコピーする。
"""

import os
import random
import shutil
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def parse_arguments():
    """
    Parse command-line arguments for the problem sampling script.
    
    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Randomly select and copy problems from source to destination directory.")
    parser.add_argument("src_root", type=str, help="Source root directory containing problem files.")
    parser.add_argument("dest_dir", type=str, help="Destination directory to copy selected problems.")
    parser.add_argument("-n", "--num_problems", type=int, default=100, help="Number of problems to select (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without copying files.")
    return parser.parse_args()

def compute_file_hash(file_path):
    """
    Compute the MD5 hash of a file's content to identify duplicates.
    Parameters:
        file_path (str): The path to the file for which to compute the hash.
    Returns:
        str: The MD5 hash of the file's content.
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def collect_json_files(src_root):
    """
    Recursively collect all JSON files from the source root directory.
    Parameters:
        src_root (str): The root directory to search for JSON files.
    Returns:
        list[str]: A list of file paths to JSON files found in the source directory.
    """
    json_files = []
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def select_unique_problems(json_files, num_problems, seed):
    """
    Select a specified number of unique problems from the list of JSON files by computing their content hashes.
    Parameters:
        json_files (list[str]): A list of file paths to JSON files to select from.
        num_problems (int): The number of unique problems to select.
        seed (int): The random seed for reproducibility.
    Returns:
        list[str]: A list of file paths to the selected unique JSON files.
    """
    hash_to_file = {}
    for file in json_files:
        file_hash = compute_file_hash(file)
        if file_hash not in hash_to_file:
            hash_to_file[file_hash] = file

    unique_files = list(hash_to_file.values())
    if len(unique_files) < num_problems:
        print(f"Warning: Only {len(unique_files)} unique problems available, less than requested {num_problems}.")
        num_problems = len(unique_files)

    random.seed(seed)
    selected_files = random.sample(unique_files, num_problems)
    return selected_files

def make_subdirectory(dest_dir, sub_dir_name):
    """
    Create a subdirectory within the destination directory to store copied files.
    Parameters:
        dest_dir (str): The main destination directory where the subdirectory will be created.
        sub_dir_name (str): The name of the subdirectory to create (e.g., "run_20251012-1630").
    Returns:
        str: The path to the created subdirectory."""
    sub_dir_path = os.path.join(dest_dir, sub_dir_name)
    os.makedirs(sub_dir_path, exist_ok=True)
    return sub_dir_path

def copy_files(selected_files, dest_dir, dry_run):
    """
    Copy the selected files to the destination directory, organizing them into a timestamped subdirectory. If dry_run is True, only print the actions without copying.
    Parameters:
        selected_files (list[str]): A list of file paths to the selected JSON files to copy.
        dest_dir (str): The main destination directory where the files will be copied.
        dry_run (bool): If True, do not actually copy files, just print the intended actions.
    Returns:
        list[str]: A list of file paths to the copied files in the destination directory (empty if dry_run is True).
    """
    copied_files = []

    for file in tqdm(selected_files, desc="Copying files"):
        parent_dir = Path(file).parent.name
        filename = Path(file).name
        new_filename = f"{Path(filename).stem}_{parent_dir}{Path(filename).suffix}"

        if dry_run:
            print(f"[Dry Run] Would copy {file} to {dest_dir}")
        else:
            # 新しいサブディレクトリ名を生成（例：run_20251012-1630）
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            sub_dir_name = f"run_{timestamp}"

            # dest_dir/ の下に新しいフォルダを作成
            run_dir = make_subdirectory(dest_dir, sub_dir_name)

            dest_path = os.path.join(run_dir, new_filename)
            shutil.copy2(file, dest_path)
            copied_files.append(dest_path)
            print(f"Copied {file} to {dest_path}")
    
    return copied_files


def main():
    args = parse_arguments()

    print(f"Source Directory: {args.src_root}")
    print(f"Destination Directory: {args.dest_dir}")
    print(f"Number of Problems to Select: {args.num_problems}")
    print(f"Random Seed: {args.seed}")
    print(f"Dry Run Mode: {'Enabled' if args.dry_run else 'Disabled'}")

    json_files = collect_json_files(args.src_root)
    print(f"Total JSON files found: {len(json_files)}")

    selected_files = select_unique_problems(json_files, args.num_problems, args.seed)
    print(f"Unique problems selected: {len(selected_files)}")

    copied_files = copy_files(selected_files, args.dest_dir, args.dry_run)

    print("\nSummary:")
    print(f"Total JSON files found: {len(json_files)}")
    print(f"Unique problems selected: {len(selected_files)}")
    if not args.dry_run:
        print(f"Files copied to: {os.path.join(args.dest_dir, f'run_{datetime.now().strftime('%Y%m%d-%H%M')}')}")
        print(f"Total files copied: {len(copied_files)}")
    else:
        print("Dry run mode - no files were copied.") 
    print("Operation completed.")

if __name__ == "__main__":
    main()


