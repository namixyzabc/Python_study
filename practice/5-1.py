from pathlib import Path
from shutil import move

path = Path(r'C:\Users\test\Python_study\test_folder\a_dir')

target = Path(r'C:\Users\test\Python_study\test_folder\b_dir')

move(str(path),str(target))