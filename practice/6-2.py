from pathlib import Path
from datetime import datetime
path = Path(r'C:\Users\test\Python_study\test_folder\b_dir\ant.txt')

'''
datetime.today()には以下の特徴があり、これがメソッドであることを示しています：

括弧()がある - メソッドを呼び出すときは括弧を使います
実行すると何らかの処理を行う - この場合は「現在の日時を取得する」という処理
呼び出すと値を返す - 現在の日時を表す新しいdatetimeオブジェクトを返します

正確にはクラスメソッド
datetime.today()は、より正確には「クラスメソッド」と呼ばれる種類のメソッドです。

通常のメソッド：オブジェクトのインスタンスに対して呼び出す（例：my_file.read()）
クラスメソッド：クラス自体に対して呼び出す（例：datetime.today()）
'''

today = datetime.today()
print(today)