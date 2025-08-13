from pathlib import Path
current = Path(r'C:\Users\test\Python_study')

'''
stemは英語の「幹」や「茎」を意味する単語です。
ファイル名から拡張子を除いた部分を表します

'*.*'は、任意の名前と任意の拡張子を持つすべてのファイルに一致するパターンです。
つまり、ディレクトリ内のすべてのファイルを検索します。
'''


for path in current.glob('*.*'):
 #   print(path.suffix)
    print(path.stem)
    