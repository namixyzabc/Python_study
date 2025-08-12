
### 1. クラスとは何か？

クラスとは、**オブジェクトの「設計図」**です。オブジェクトとは、データ（**属性**）とそのデータを操作するための手続き（**メソッド**）をひとまとめにしたものです。

#### なぜクラスが必要か？
プログラミングでは、関連するデータと処理をまとめて管理したい場面がよくあります。例えば、「ユーザー」を表現する場合、`名前`、`メールアドレス`といったデータと、`ログインする`、`名前を変更する`といった処理が必要です。

クラスを使わない場合、これらはバラバラに管理することになります。
```python
user1_name = "Taro"
user1_email = "taro@example.com"

def login(name):
    print(f"{name}がログインしました。")

login(user1_name)
```
これでは、ユーザーが増えるたびに変数を増やさなければならず、管理が大変です。

クラスを使うと、これらのデータと処理を「User」という一つのまとまりとして定義できます。この設計図（クラス）を元に、TaroさんやHanakoさんといった具体的なユーザー（**インスタンス**）を作成できます。

- **クラス (Class)**: 設計図。例えば「人間」の定義。
- **インスタンス (Instance)**: 設計図を元に作られた実物。オブジェクトとも呼ばれます。例えば「太郎さん」という具体的な人間。
- **属性 (Attribute)**: インスタンスが持つデータ。例えば「名前」「年齢」。
- **メソッド (Method)**: インスタンスが行う操作（処理）。例えば「歩く」「話す」。

このように、モノ（オブジェクト）を単位としてプログラミングを行う考え方を**オブジェクト指向プログラミング (Object-Oriented Programming, OOP)**と呼びます。

### 2. クラスの基本

#### 2.1. クラスの定義とインスタンスの作成
`class` キーワードを使ってクラスを定義します。クラス名は大文字で始めるのが慣習です（`CamelCase`）。

```python
# 'User'というクラス（設計図）を定義
class User:
    pass  # 中身が空の場合はpassと書く

# クラスからインスタンス（実物）を作成
user1 = User() 
user2 = User()

print(type(user1))  # <class '__main__.User'>
print(user1)        # <__main__.User object at 0x...> メモリアドレスが表示される
```

#### 2.2. コンストラクタ (`__init__`) とインスタンス変数
インスタンスが作成されるときに**自動的に呼び出される特別なメソッド**がコンストラクタです。Pythonでは `__init__` という名前で定義します。

ここで、各インスタンスが個別に持つデータ（**インスタンス変数**）を初期設定します。インスタンス変数は `self.変数名` の形で定義します。

- **`self`**: 作成されたインスタンス自身を指す特別な引数です。メソッドを定義する際の最初の引数には、必ず `self` を書きます（呼び出す際には不要です）。

```python
class User:
    # コンストラクタ
    def __init__(self, name, email):
        print(f"{name}さんのインスタンスを作成します。")
        # インスタンス変数（属性）を定義
        self.name = name
        self.email = email

# インスタンス作成時に__init__が呼ばれ、引数が渡される
user1 = User("Taro", "taro@example.com")
user2 = User("Hanako", "hanako@example.com")

# 各インスタンスはそれぞれ独立したデータを持つ
print(user1.name)   # Taro
print(user2.name)   # Hanako
```

#### 2.3. インスタンスメソッド
インスタンスが持つデータを操作するための関数が**インスタンスメソッド**です。第一引数に `self` を取り、インスタンス変数にアクセスできます。

```python
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    # インスタンスメソッドの定義
    def greet(self):
        # selfを使ってインスタンス変数にアクセス
        print(f"こんにちは！私の名前は{self.name}です。")

user1 = User("Taro", "taro@example.com")

# メソッドの呼び出し
user1.greet()  # こんにちは！私の名前はTaroです。
```

### 3. クラス変数と各種メソッド

#### 3.1. クラス変数
インスタンス変数 (`self.name`など)が各インスタンスに固有のデータだったのに対し、**クラス変数はすべてのインスタンスで共有されるデータ**です。クラス定義の直下に書きます。

```python
class Product:
    # クラス変数 (消費税率)
    tax_rate = 0.10

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def get_price_with_tax(self):
        # クラス変数は self.変数名 または クラス名.変数名 でアクセス可能
        return self.price * (1 + Product.tax_rate)

apple = Product("りんご", 120)
orange = Product("オレンジ", 100)

print(apple.get_price_with_tax())   # 132.0
print(orange.get_price_with_tax())  # 110.0

# クラス変数を変更すると、すべてのインスタンスに影響する
Product.tax_rate = 0.08
print(apple.get_price_with_tax())   # 129.6
```

#### 3.2. クラスメソッドとスタティックメソッド
インスタンスメソッドの他に、特殊なメソッドがあります。

- **クラスメソッド (`@classmethod`)**
  - インスタンスではなく、**クラス自体を操作するためのメソッド**です。
  - 第一引数にインスタンス(`self`)の代わりに**クラス(`cls`)**を取ります。
  - **`@classmethod`** という**デコレータ**を付けて定義します。
  - 主に、クラス変数を使った処理や、特定の形式からインスタンスを生成する（ファクトリメソッド）場合に使われます。

- **スタティックメソッド (`@staticmethod`)**
  - **クラスやインスタンスの状態に依存しない、単なる関数**をクラス内に置きたい場合に使います。
  - 引数に `self` も `cls` も取りません。
  - **`@staticmethod`** というデコレータを付けて定義します。
  - クラスの名前空間（クラスに属する領域）に配置したいヘルパー関数などに便利です。

```python
class MyClass:
    class_var = "クラス変数"

    def __init__(self, instance_var):
        self.instance_var = instance_var

    # インスタンスメソッド
    def instance_method(self):
        print(f"インスタンスメソッドです。{self.instance_var}にアクセスできます。")

    # クラスメソッド
    @classmethod
    def class_method(cls):
        print(f"クラスメソッドです。{cls.class_var}にアクセスできます。")

    # スタティックメソッド
    @staticmethod
    def static_method(arg1, arg2):
        print("スタティックメソッドです。クラスやインスタンスの状態には依存しません。")
        return arg1 + arg2

# インスタンスメソッドはインスタンスから呼び出す
obj = MyClass("インスタンス変数")
obj.instance_method()

# クラスメソッドはクラスからもインスタンスからも呼び出せる
MyClass.class_method()
obj.class_method() # こちらも可能

# スタティックメソッドも同様
MyClass.static_method(10, 20)
obj.static_method(10, 20) # こちらも可能
```

| メソッドの種類 | 第一引数 | アクセスできる変数 | デコレータ | 主な用途 |
| :--- | :--- | :--- | :--- | :--- |
| インスタンスメソッド | `self` | インスタンス変数, クラス変数 | なし | インスタンスの状態を操作 |
| クラスメソッド | `cls` | クラス変数のみ | `@classmethod` | クラス変数の操作、ファクトリ |
| スタティックメソッド | なし | なし | `@staticmethod` | クラスに属するユーティリティ関数 |

### 4. 継承 (Inheritance)

**継承**とは、**既存のクラス（親クラス）の機能を引き継いで、新しいクラス（子クラス）を作成する仕組み**です。コードの再利用性を高め、拡張を容易にします。

- **親クラス (Superclass/Parent class)**: 継承元となるクラス。
- **子クラス (Subclass/Child class)**: 継承して作られた新しいクラス。

#### 4.1. 継承の基本とメソッドのオーバーライド
子クラスは親クラスの属性やメソッドをすべて受け継ぎます。また、子クラスで親クラスと同じ名前のメソッドを定義することで、処理を上書き（**オーバーライド**）できます。

```python
# 親クラス
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("何かの声で鳴きます")

# 子クラス (Animalを継承)
class Dog(Animal):
    # speakメソッドをオーバーライド
    def speak(self):
        print(f"{self.name}は「ワン！」と鳴きます")

class Cat(Animal):
    # speakメソッドをオーバーライド
    def speak(self):
        print(f"{self.name}は「ニャー」と鳴きます")

# インスタンスを作成
animal = Animal("動物")
pochi = Dog("ポチ")
tama = Cat("タマ")

animal.speak() # 何かの声で鳴きます
pochi.speak()  # ポチは「ワン！」と鳴きます
tama.speak()   # タマは「ニャー」と鳴きます
```

#### 4.2. `super()` による親クラスのメソッド呼び出し
子クラスで親クラスのメソッドをオーバーライドしつつ、**親クラスの処理も呼び出したい**場合があります。その際に `super()` を使います。

特に、子クラスの `__init__` で親クラスの `__init__` を呼び出し、親クラスの属性を初期化するのは非常によくあるパターンです。

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"私の名前は{self.name}、{self.age}歳です。")

class Employee(Person):
    def __init__(self, name, age, employee_id):
        # super()で親クラスの__init__を呼び出す
        super().__init__(name, age)
        # 子クラス独自の属性を追加
        self.employee_id = employee_id

    # greetメソッドをオーバーライドしつつ、親の処理も使う
    def greet(self):
        super().greet() # 親のgreetメソッドを呼び出す
        print(f"社員番号は{self.employee_id}です。")

# インスタンスを作成
employee = Employee("Suzuki", 30, "E12345")
employee.greet()
# 出力:
# 私の名前はSuzuki、30歳です。
# 社員番号はE12345です。
```

### 5. カプセル化 (Encapsulation)

**カプセル化**とは、データ（属性）とそれを操作するメソッドをクラス内にまとめ、**外部から直接データにアクセスさせないようにする**考え方です。これにより、意図しないデータの書き換えを防ぎ、クラスの安全性を高めます。

Pythonには厳密なアクセス制限機能はありませんが、変数名の命名規則によってアクセスレベルを示す慣習があります。

- **`public` (公開)**: 普通の変数名 (例: `name`)。どこからでもアクセス可能です。
- **`_protected` (保護)**: アンダースコア1つ (例: `_name`)。「クラス内部と子クラスからのみアクセスしてください」という開発者間の紳士協定です。
- **`__private` (非公開)**: アンダースコア2つ (例: `__name`)。クラスの外部から直接アクセスしようとするとエラーになります。これは**名前マングリング**という仕組みで、`_クラス名__変数名` のように名前が自動的に変換されるためです。

```python
class Wallet:
    def __init__(self, initial_amount):
        # 非公開にしたい属性はアンダースコア2つで始める
        self.__balance = initial_amount

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"{amount}円入金しました。")
        else:
            print("不正な金額です。")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"{amount}円出金しました。")
        else:
            print("残高が不足しています。")

    def get_balance(self):
        return self.__balance

my_wallet = Wallet(1000)

# 直接残高を書き換えようとしてもできない
# my_wallet.__balance = -5000 # これはエラーにはならないが、新しい属性が作られるだけ
# print(my_wallet.__balance) # AttributeError: 'Wallet' object has no attribute '__balance'

# 必ずメソッドを通して操作する
my_wallet.deposit(500)
print(f"現在の残高: {my_wallet.get_balance()}円") # 現在の残高: 1500円
```

### 6. プロパティ (`@property`)

カプセル化で属性を非公開にすると、値の取得や設定に `get_...`, `set_...` のようなメソッドが必要になり、コードが少し冗長になります。**プロパティ**を使うと、**メソッドを属性のように（`()`なしで）扱う**ことができます。

`@property` デコレータは、値を取得するメソッド（**ゲッター**）を定義します。
`@<属性名>.setter` デコレータは、値を設定するメソッド（**セッター**）を定義します。

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius # 保護された属性

    # ゲッター: c.radius のようにアクセスできる
    @property
    def radius(self):
        print("半径を取得します。")
        return self._radius

    # セッター: c.radius = 10 のように代入できる
    @radius.setter
    def radius(self, value):
        print("半径を設定します。")
        if value <= 0:
            raise ValueError("半径は正の値でなければなりません。")
        self._radius = value

    # ゲッターのみ定義されたプロパティ
    @property
    def area(self):
        # 面積は半径から計算されるため、セッターは不要
        return self._radius ** 2 * 3.14

c = Circle(5)

# ゲッターが呼ばれる (メソッドだが()は不要)
print(f"半径: {c.radius}") # 半径: 5

# セッターが呼ばれる
c.radius = 10 
print(f"新しい半径: {c.radius}") # 新しい半径: 10

# 計算されたプロパティにアクセス
print(f"面積: {c.area}") # 面積: 314.0

# 不正な値を設定しようとするとエラー
try:
    c.radius = -1
except ValueError as e:
    print(e) # 半径は正の値でなければなりません。
```

### 7. 特殊メソッド (Special/Magic Methods)

`__init__` や `__str__` のように、アンダースコア2つで囲まれたメソッドを**特殊メソッド**（または**マジックメソッド**）と呼びます。これらをクラスに定義することで、Pythonの組み込み演算子や関数（`+`, `len()`, `print()`など）の挙動をカスタマイズできます。

- `__str__(self)`: `print(instance)` や `str(instance)` で呼び出される。**人間が読みやすい**文字列を返すのが目的。
- `__repr__(self)`: `repr(instance)` や対話モードでインスタンスを評価したときに呼び出される。**開発者向け**で、そのオブジェクトを再現できるような文字列を返すのが理想。
- `__len__(self)`: `len(instance)` で呼び出される。
- `__eq__(self, other)`: `instance1 == instance2` で呼び出される。

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    # print()で表示される形式を定義
    def __str__(self):
        return f"『{self.title}』(著者: {self.author})"

    # オブジェクトそのものを表す形式を定義
    def __repr__(self):
        return f"Book(title='{self.title}', author='{self.author}', pages={self.pages})"

    # len()でページ数を返すように定義
    def __len__(self):
        return self.pages

book = Book("Python入門", "Taro Yamada", 300)

print(book)          # __str__が呼ばれる: 『Python入門』(著者: Taro Yamada)
print(str(book))     # __str__が呼ばれる: 『Python入門』(著者: Taro Yamada)
print(repr(book))    # __repr__が呼ばれる: Book(title='Python入門', author='Taro Yamada', pages=300)
print(len(book))     # __len__が呼ばれる: 300
```

### 8. 応用的なトピック

#### 8.1. 多重継承とMRO
Pythonは、複数の親クラスから継承する**多重継承**をサポートしています。

```python
class Father:
    def skill_f(self):
        print("父のスキル")

class Mother:
    def skill_m(self):
        print("母のスキル")

# 2つのクラスを継承
class Child(Father, Mother):
    pass

c = Child()
c.skill_f() # 父のスキル
c.skill_m() # 母のスキル
```

もし複数の親クラスに同じ名前のメソッドがあった場合、どのメソッドが呼ばれるかは**MRO (Method Resolution Order, メソッド解決順序)**に従います。MROは `クラス名.mro()` で確認できます。基本的には、継承時に指定したクラスの左側が優先されます。

#### 8.2. 抽象クラス (Abstract Base Class)
**抽象クラス**とは、**インスタンス化できない、継承されることを前提としたクラス**です。子クラスに特定のメソッドの実装を強制したい場合（＝インターフェースの定義）に使います。

`abc`モジュールと`@abstractmethod`デコレータを使います。

```python
from abc import ABC, abstractmethod

# ABCを継承して抽象クラスを定義
class Shape(ABC):
    @abstractmethod
    def area(self):
        # 処理は書かずにpassする
        pass

# 抽象クラスを継承した子クラス
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    # 抽象メソッドを必ず実装（オーバーライド）する必要がある
    def area(self):
        return self.width * self.height

# Shapeはインスタンス化できない
# s = Shape() # TypeError: Can't instantiate abstract class Shape with abstract method area

# Rectangleはareaを実装しているのでインスタンス化できる
r = Rectangle(10, 5)
print(r.area()) # 50
```
抽象クラスを使うことで、「`Shape`を継承するクラスは、必ず`area`メソッドを持たなければならない」というルールをプログラム上で強制できます。
