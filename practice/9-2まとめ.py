from pathlib import Path
txt = 'testプログラム'
path = Path('replist.csv')
csvtxt = path.read_text(encoding='utf-8')
lst = csvtxt.splitlines()  # 文字列を行ごとに分割してリストに格納

for line in lst:
    words = line.split(',')  #行をカンマで分割し、リスト形式で変数wordsに格納します。
    txt = txt.replace(words[0],words[1])  #words[0]は置換前の文字列（検索文字列）。words[1]は置換後の文字列
    print(txt)