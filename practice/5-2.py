from pathlib import Path
from shutil import copy

path = Path(r'C:\Users\test\Python_study\test_folder\ant.TXT')

target = Path(r'C:\Users\test\Python_study\test_folder\b_dir')

copy(str(path),str(target))