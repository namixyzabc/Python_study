from pathlib import Path
from shutil import copy

current = Path(r'C:\Users\test\Python_study\test_folder')
for path in current.glob('*.*'):
    if path.match('*.py'):
        continue
    ext = path.suffix[1:]
    #target = Path(ext)  #コードの実行時のカレントディレクトリに対して行われるため、注意
    target = Path(r'C:\Users\test\Python_study\test_folder') / ext

    target.mkdir(exist_ok=True)
    copy(str(path),str(target))

r'''
/ 演算子:
Path オブジェクトで / 演算子を使うと、パスの連結が行われます
これは通常のフォルダ区切り文字（Windows なら \、Mac/Linux なら /）の働きをします
プログラムが動作するOSに関係なく同じ書き方でパスを連結できる便利な機能です

ext:
ファイルの拡張子（例：「txt」「jpg」「pdf」など）が入っている変数です
前のコードで ext = path.suffix[1:] として取得したものです
結果として target には C:\Users\test\Python_study\test_folder\txt という Path オブジェクトが格納されます
'''
