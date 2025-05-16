lst = ['春', '秋', '夏', '冬', ]

item = lst.pop(2)  #pop()メソッドは、データからアイテムを取り出しながら同時に削除するメソッド
print(lst)
print(item)

lst2 = ['春', '秋', '夏', '冬', ]
for item in reversed(lst2):
    print(item)