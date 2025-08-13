from pathlib import Path

# 指定したディレクトリ内の「テキスト1.txt」というファイルへのパスオブジェクトを作成しています。
path = Path(r'C:\Users\test\Python_study\test_folder\テキスト1.txt')

#パスオブジェクトの read_text() メソッドを使って、指定したファイルの内容をテキストとして読み込みます。
txt = path.read_text(encoding='utf-8')


print(txt)