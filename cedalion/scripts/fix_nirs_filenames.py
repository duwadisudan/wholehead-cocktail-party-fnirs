"""
Script to fix NIRS file naming discrepancies.

This script corrects filenames in sub-XX/nirs folders where the filename
contains a different subject number than the parent folder.

Example:
    sub-30/nirs/sub-31_task-control_run-01_events.tsv
    -> sub-30/nirs/sub-30_task-control_run-01_events.tsv

Usage:
    python fix_nirs_filenames.py <path_to_master_data_folder> [--dry-run]
"""

import os
import re
import argparse
from pathlib import Path


def extract_subject_number(folder_name):
    """Extract subject number from folder name like 'sub-31'."""
    match = re.match(r'sub-(\d+)', folder_name)
    if match:
        return match.group(1)
    return None


def fix_filenames_in_nirs_folder(master_data_path, dry_run=True):
    """
    Fix filenames in nirs folders to match their parent subject folder.
    
    Args:
        master_data_path: Path to the master data folder containing sub-XX folders
        dry_run: If True, only print what would be changed without actually renaming
    
    Returns:
        Dictionary with statistics about changes made
    """
    master_path = Path(master_data_path)
    
    if not master_path.exists():
        print(f"Error: Path does not exist: {master_data_path}")
        return None
    
    stats = {
        'total_files_checked': 0,
        'files_renamed': 0,
        'errors': 0,
        'changes': []
    }
    
    # Find all sub-XX folders
    subject_folders = sorted([f for f in master_path.iterdir() 
                             if f.is_dir() and f.name.startswith('sub-')])
    
    if not subject_folders:
        print(f"No subject folders found in {master_data_path}")
        return stats
    
    print(f"Found {len(subject_folders)} subject folders")
    print("-" * 80)
    
    for subject_folder in subject_folders:
        correct_subject_num = extract_subject_number(subject_folder.name)
        
        if not correct_subject_num:
            print(f"Warning: Could not extract subject number from {subject_folder.name}")
            continue
        
        nirs_folder = subject_folder / 'nirs'
        
        if not nirs_folder.exists():
            continue
        
        # Check all files in the nirs folder
        for file_path in nirs_folder.iterdir():
            if not file_path.is_file():
                continue
            
            stats['total_files_checked'] += 1
            filename = file_path.name
            
            # Check if filename starts with sub-XX pattern
            match = re.match(r'sub-(\d+)(_.*)', filename)
            
            if match:
                file_subject_num = match.group(1)
                remainder = match.group(2)
                
                # If the subject number in filename doesn't match the folder
                if file_subject_num != correct_subject_num:
                    new_filename = f"sub-{correct_subject_num}{remainder}"
                    new_path = file_path.parent / new_filename
                    
                    # Check if target file already exists
                    if new_path.exists():
                        print(f"⚠️  WARNING: Target file already exists!")
                        print(f"   {file_path}")
                        print(f"   -> {new_path}")
                        print(f"   Skipping to avoid overwrite.\n")
                        stats['errors'] += 1
                        continue
                    
                    change_info = {
                        'folder': subject_folder.name,
                        'old_name': filename,
                        'new_name': new_filename,
                        'full_old_path': str(file_path),
                        'full_new_path': str(new_path)
                    }
                    
                    stats['changes'].append(change_info)
                    
                    if dry_run:
                        print(f"📝 Would rename in {subject_folder.name}/nirs/:")
                        print(f"   {filename}")
                        print(f"   -> {new_filename}\n")
                    else:
                        try:
                            file_path.rename(new_path)
                            print(f"✅ Renamed in {subject_folder.name}/nirs/:")
                            print(f"   {filename}")
                            print(f"   -> {new_filename}\n")
                            stats['files_renamed'] += 1
                        except Exception as e:
                            print(f"❌ Error renaming {file_path}: {e}\n")
                            stats['errors'] += 1
    
    return stats


def print_summary(stats, dry_run):
    """Print summary of changes."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if stats:
        print(f"Total files checked: {stats['total_files_checked']}")
        print(f"Files with naming issues: {len(stats['changes'])}")
        
        if dry_run:
            print(f"Files that would be renamed: {len(stats['changes'])}")
        else:
            print(f"Files successfully renamed: {stats['files_renamed']}")
            print(f"Errors encountered: {stats['errors']}")
        
        if stats['changes']:
            print("\nDetailed changes:")
            for i, change in enumerate(stats['changes'], 1):
                print(f"\n{i}. Folder: {change['folder']}")
                print(f"   Old: {change['old_name']}")
                print(f"   New: {change['new_name']}")
    
    if dry_run:
        print("\n💡 This was a DRY RUN. No files were actually renamed.")
        print("   Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description='Fix NIRS file naming discrepancies in subject folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without making them
  python fix_nirs_filenames.py "U:\\path\\to\\master_data" --dry-run
  
  # Actually rename the files
  python fix_nirs_filenames.py "U:\\path\\to\\master_data"
        """
    )
    
    parser.add_argument(
        'master_data_path',
        type=str,
        help='Path to the master data folder containing sub-XX folders'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Preview changes without actually renaming files (default: True)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually rename files (overrides --dry-run)'
    )
    
    args = parser.parse_args()
    
    # If --execute is specified, turn off dry_run
    dry_run = args.dry_run and not args.execute
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be renamed\n")
    else:
        print("⚠️  EXECUTE MODE - Files will be renamed!\n")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    print(f"Scanning folder: {args.master_data_path}\n")
    
    stats = fix_filenames_in_nirs_folder(args.master_data_path, dry_run=dry_run)
    print_summary(stats, dry_run)


if __name__ == "__main__":
    main()
