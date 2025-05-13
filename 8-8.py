import re
txt = '12112312-00213123'
if re.match(r'\d{3}-\d{4}', txt):
    print(f'{txt}は郵便番号')
else:
    print(f'{txt}郵便番号ではない')
