from pathlib import Path
path = Path('sample.txt')
txt = path.read_text(encoding='utf-8')
txt = txt.replace('123','modifi')
print(txt)



