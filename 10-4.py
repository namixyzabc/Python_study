from itertools import product

#product関数は、Pythonのitertoolsモジュールに含まれる便利な関数で、複数のイテラブル（リスト、タプル、文字列など）のデカルト積（直積）を生成します。

'''
イテラブル（Iterable）とは、要素を一つずつ返すことができるオブジェクトのことです。簡単に言えば、「繰り返し処理ができるオブジェクト」や「for文で回せるオブジェクト」と考えることができます。
リスト（list）：[1, 2, 3]
タプル（tuple）：(1, 2, 3)
文字列（str）："hello"
辞書（dict）：{"a": 1, "b": 2}
集合（set）：{1, 2, 3}
ファイルオブジェクト
rangeオブジェクト：range(10)
ジェネレータオブジェクト
'''
for x, y in product(range(1, 10), range(1, 10)):
    print(f'{x}*{y}={x*y}')