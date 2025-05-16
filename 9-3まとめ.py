from pathlib import Path
from shutil import copy
current = Path()

for path in current.glob('*.*'):
    if path.match('*.py'):
        continue
    ext = path.suffix # 例: '.txt'
    ext = ext.lower()[1:]  # 小文字化して先頭の'.'を除去 → 'txt'
    target = Path(ext)#"txt"（なら） という名前の★ディレクトリを表す Path オブジェクトになる
    target.mkdir(exist_ok=True)
    copy(str(path), str(target))#example.txt を txt ディレクトリにコピーします
    tpath = target / path #example.txt の場合、tpath は txt/example.txt
    npath = target / Path(path.stem + '.' + ext) #
    tpath.rename(npath)

