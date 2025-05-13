from pathlib import Path

#exist_ok=True: このオプションを指定すると、
# 既にディレクトリが存在している場合でもエラーを発生させずに処理を続行します。
path = Path(r'C:\Users\test\Python_study\test_folder\aaadir')
path.rmdir()