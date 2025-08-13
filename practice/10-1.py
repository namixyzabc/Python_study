lst = ['春', '秋', '夏', '冬', ]

for index,  item in enumerate(lst):
    print(index, item)

lst.append('年末')
print(lst)

lst.remove('夏')
print(lst)

'''
キーワードとは、プログラミング言語の構文を構成する特別な単語です。

特徴：
言語の文法を形成する基本的な要素
プログラミング言語によって予め定義され、特別な意味を持つ
変数名や関数名として使用できない（予約されている）
呼び出すものではなく、構文の一部として使用する

Pythonのキーワード例
if, else, for, while, def, class, return, import, del, in, is, pass, break, continue など
'''
lst2 = ['春', '秋', '夏', '冬', ]
del lst2[0]
print(lst2)