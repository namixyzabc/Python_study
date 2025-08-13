from pathlib import Path
current = Path(r'C:\Users\test\Python_study\test_folder')



'''
target = f'{path.stem}.TXT'　とすると、フォルダパスを含まないファイル名になってしまい、
path.rename(target)  が実行されたときに、カレントディレクトリに対して実行しようとして、
エラーになる。

★path.with_name()は、Pathオブジェクトのメソッドで、ファイルのパス（場所）はそのままに、
　ファイル名だけを変更するための便利な機能です。

'''
for path in current.glob('*.txt'):
    target = path.with_name(f'{path.stem}.TXT')
    path.rename(target)   