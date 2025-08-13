from pathlib import Path

current = Path()
target = Path('result')
target.mkdir(exist_ok=True)

for path in current.glob('*.txt'):
    txt = path.read_text(encoding='utf-8')
    txt = txt.replace('あいうえお','パイソン')
    tpath = target / path
    tpath.write_text(txt, encoding='utf-8')
