import re
txt = 'aweaf121-3123oaiwht'
lst = re.findall(r'\d{3}-\d{4}', txt)
print(lst)
