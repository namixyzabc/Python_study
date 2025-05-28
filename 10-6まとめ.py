from pathlib import Path
dct = {}
current = Path()
for path in current.glob('**/*.*'):  #カレントディレクトリ内のすべてのファイルを再帰的に走査
    ext = path.suffix
    if ext in dct:  #もし辞書dctにこの拡張子がすでに存在していれば、その値に1を加えます。存在していなければ、その拡張子をキーとして、値を1として辞書に追加します。ls
        dct[ext] += 1  #もし辞書に既にその拡張子があれば、カウントを1増やします
    else:
        dct[ext] = 1
print(dct)

'''
dct[ext]の部分の解説
# 例えば、辞書の中身がこうなっているとします
dct = {'.txt': 3, '.py': 2, '.jpg': 5}

# そのとき ext が '.txt' だとすると
ext = '.txt'
dct[ext]  # これは 3 を返します
'''