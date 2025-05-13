
'''
pathlib.Pathオブジェクトの代わりに、文字列としてファイルパスを直接使用
open()関数でファイルを書き込みモード('w')で開く
元のコードと同じく'utf-8'エンコーディングを指定

Pythonのwith文は、リソースの管理を簡単にするための構文です。主にファイル操作
やネットワーク接続などで使われ、リソースの解放を自動的に行います。

with文を使用してファイルを開き、処理後に自動的に閉じるようにしています
with文を使うことでファイルのオープンとクローズを自動的に処理します。
with文を使うことで、エラーが発生してもリソースが適切に解放される
'''


path = r'C:\Users\test\Python_study\test_folder\テキスト2.txt'
txt = 'aiuawdawa132123deo641924691867aaaaaa\nwaedawdawd/naw8otyao1372031kkkkkkkkkkkkkkkkk231'

'''
as f: ファイルオブジェクトを変数fに代入します。
'''
with open(path, 'a', encoding='utf-8') as f:
    f.write(txt)