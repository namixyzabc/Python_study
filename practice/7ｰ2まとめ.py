from pathlib import Path
from shutil import move
from datetime import datetime

current = Path(r'C:\Users\test\Python_study\test_folder')
target = Path(r'C:\Users\test\Python_study\test_folder') / 'old'
target.mkdir(exist_ok=True)
today = datetime.today()
untilday = datetime(today.year,today.month,1)

for path in current.glob('*.*'):
    st_mtime = path.stat().st_mtime
    update = datetime.fromtimestamp(st_mtime)
    if update < untilday:
        move(str(path),str(target))