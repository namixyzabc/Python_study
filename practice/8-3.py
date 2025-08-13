from pathlib import Path
path = Path('sample.txt')

txt = 'ABC/def/GHI'
lst = txt.split('/')
print(lst)

txt2 = path.read_text(encoding='utf-8')
lst2 = txt2.splitlines()
for line in lst2:
    adline = line + 'qwe'
    print(adline)