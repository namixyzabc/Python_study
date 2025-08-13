from pathlib import Path
path = Path('テキスト1.TXT')
txt = path.read_text(encoding='utf-8')
cnt = txt.count('あいうえお')
print(cnt)