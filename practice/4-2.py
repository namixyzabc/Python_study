from pathlib import Path
current = Path(r'C:\Users\test\Python_study')

#　globは「global」の略で、広範囲にわたる検索を意味します。

for path in current.glob('*'):
    if path.is_dir():
        print(f'{path} フォルダです')
    else:
        print(f'{path} フォルダではありません')
    