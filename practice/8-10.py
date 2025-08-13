import re


'''
"substitute"（置換）の略
(): パターン全体を括弧で囲むことでグループ化
\1: 最初の括弧でグループ化したパターン（郵便番号）を参照
'''

txt = '郵便番号は120-1119と130-1111'
txt = re.sub(r'(\d{3}-\d{4})',r'『\1』', txt)
print(txt)

txt = re.sub(r'(\d{3})-(\d{4})',r'『\2=\1』', txt)
print(txt)
             