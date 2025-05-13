from pathlib import Path
current = Path(r'C:\Users\test\Python_study')

'''
path.match('*')の意味
このコードは、pathオブジェクトが指定されたパターンに一致するかどうかをチェックしています。
'''

for path in current.glob('*'):
    if path.match('*'):
        print(path)
    