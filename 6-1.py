from pathlib import Path
from datetime import datetime
path = Path(r'C:\Users\test\Python_study\test_folder\b_dir\ant.txt')


'''
path.stat() は、指定したファイル
（このケースでは path 変数に格納されているファイルパス）の統計情報を取得するメソッドです。

st_mtime は、「status time modification time」の略で、
ファイルの内容が最後に変更された時刻を示します。

UNIXタイムスタンプとは、1970年1月1日午前0時（UTC）からの経過秒数を表す数値です。
例えば：1609459200.0 というタイムスタンプは 2021年1月1日 0:00:00 を表します

＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
↓変数　　　　　　　　　　　↓属性　　　　
st_mtime = path.stat().st_mtime を分解すると：
まず path.stat() というメソッドを呼び出して、stat_resultオブジェクトを取得
次に、そのオブジェクトの st_mtime という属性にアクセスしている
属性なので、値を取得するだけで何か処理を実行するわけではありません。
すでにstat()メソッドが収集した情報にアクセスしているだけです。
＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
'''

st_mtime = path.stat().st_mtime
update = datetime.fromtimestamp(st_mtime)
print(update)