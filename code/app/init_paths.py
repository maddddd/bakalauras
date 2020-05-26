import sys
import os
from pathlib import Path


def print_all_paths():
    for p in sys.path:
        print(p)


def init_sys_folders():
    code_folder_path = Path(os.getcwd()).parent
    sys.path.append(os.path.abspath(os.path.join(code_folder_path, 'cnns')))
    sys.path.append(os.path.abspath(os.path.join(code_folder_path, 'mnist')))
