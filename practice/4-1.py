from pathlib import Path
path = Path(r'C:\Users\test\Python_study\test_folder\テキスト2.txt')


'''
fの後に続く文字列の中に、{}で囲まれた部分があります。この部分に変数pathの値が埋め込まれます。

★fを使わない場合
print('{}はあります'.format(path))

OR

print(str(path) + 'はあります')

'''
if path.exists():
    print(str(path) + 'はあります')
else:
    print(f'{path}はありません')