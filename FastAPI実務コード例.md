
### 1. 基本的なCRUD APIの構築

アプリケーションの基本となる、データの作成（Create）、読み取り（Read）、更新（Update）、削除（Delete）を行うCRUD APIから始めます。まずはデータベースを使わず、メモリ上のPythonの辞書を簡易的なデータベースとして使用します。

#### 1.1. プロジェクトの準備

まず、FastAPIと、サーバーを起動するためのUvicornをインストールします。

```bash
pip install fastapi "uvicorn[standard]"
```

次に、メインのアプリケーションファイル `main.py` を作成します。

#### 1.2. Pydanticモデルの定義

FastAPIでは、リクエストとレスポンスのデータ構造をPydanticモデルを使って定義します。これにより、強力なデータバリデーションと型ヒントの恩恵を受けることができます。

今回は、シンプルな「商品（Item）」を管理するAPIを作成します。

`main.py`
```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict

# --- Pydanticモデル定義 ---

class Item(BaseModel):
    """
    商品データを表すPydanticモデル
    - id: 商品ID (オプショナル)
    - name: 商品名
    - price: 価格 (0より大きい)
    - description: 商品説明 (オプショナル)
    """
    id: Optional[int] = None
    name: str
    price: float = Field(..., gt=0, description="価格は0より大きい必要があります")
    description: Optional[str] = None

class ItemCreate(BaseModel):
    """
    商品作成時に使用するモデル (IDは自動採番のため含めない)
    """
    name: str
    price: float = Field(..., gt=0, description="価格は0より大きい必要があります")
    description: Optional[str] = None

class ItemUpdate(BaseModel):
    """
    商品更新時に使用するモデル (全てのフィールドがオプショナル)
    """
    name: Optional[str] = None
    price: Optional[float] = Field(None, gt=0, description="価格は0より大きい必要があります")
    description: Optional[str] = None


# --- アプリケーションインスタンスとDB代わりの辞書 ---

app = FastAPI()

# 簡易的なインメモリデータベース
db: Dict[int, Item] = {}
next_item_id = 0
```

**解説:**

*   `Item` モデルは、データベースに保存される完全な商品データを表します。`id` はオプショナル（`Optional`） لأنها قد لا تكون موجودة عند الإنشاء.
*   `ItemCreate` モデルは、クライアントが新しい商品を作成する際に送信するデータ構造です。IDはサーバー側で割り当てるため、このモデルには含まれていません。
*   `ItemUpdate` モデルは、商品を更新する際に使用します。ユーザーは更新したいフィールドだけを送信すればよいため、すべてのフィールドがオプショナルになっています。
*   `Field` を使うことで、`gt=0` のような詳細なバリデーションルール（0より大きい）や、APIドキュメントに表示される説明（`description`）を追加できます。

#### 1.3. CRUDエンドポイントの実装

次に、定義したモデルを使って各APIエンドポイント（パスオペレーション）を実装します。

##### **Create: 商品の新規作成**

`main.py` に以下を追加します。

```python
@app.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)
def create_item(item_create: ItemCreate) -> Item:
    """
    新しい商品を作成します。
    """
    global next_item_id
    new_id = next_item_id
    
    # ItemCreateモデルからItemモデルのインスタンスを作成
    item = Item(id=new_id, **item_create.model_dump())
    
    db[new_id] = item
    next_item_id += 1
    
    return item
```

**解説:**

*   `@app.post("/items/")`: HTTP POSTリクエストを `/items/` パスで受け付けます。
*   `response_model=Item`: レスポンスのデータ構造が `Item` モデルに準拠することを定義します。これにより、レスポンスデータが自動でバリデーションされ、ドキュメントにも明記されます。
*   `status_code=status.HTTP_201_CREATED`: 処理が成功した場合、HTTPステータスコード201を返します。
*   `item_create: ItemCreate`: リクエストボディが `ItemCreate` モデルの形式であることを示します。FastAPIが自動でJSONをパースし、バリデーションを実行します。
*   `item_create.model_dump()`: Pydanticモデルのデータを辞書に変換し、`**` を使って `Item` モデルのコンストラクタに渡しています。

##### **Read: 商品の一覧取得と個別取得**

`main.py` に以下を追加します。

```python
from typing import List

@app.get("/items/", response_model=List[Item])
def read_items() -> List[Item]:
    """
    全ての商品を取得します。
    """
    return list(db.values())

@app.get("/items/{item_id}", response_model=Item)
def read_item(item_id: int) -> Item:
    """
    指定されたIDの商品を取得します。
    """
    if item_id not in db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Item not found"
        )
    return db[item_id]
```

**解説:**

*   `@app.get(...)`: HTTP GETリクエストを受け付けます。
*   `/items/{item_id}`: パスパラメータ `item_id` を受け取ることを示します。関数の引数 `item_id: int` で型が指定され、自動で変換とバリデーションが行われます。
*   `response_model=List[Item]`: 複数の `Item` オブジェクトをリストで返すことを示します。
*   `HTTPException`: FastAPIが提供する例外クラスです。これを発生させると、適切なHTTPエラーレスポンスがクライアントに返されます。

##### **Update: 商品の更新**

`main.py` に以下を追加します。

```python
@app.put("/items/{item_id}", response_model=Item)
def update_item(item_id: int, item_update: ItemUpdate) -> Item:
    """
    指定されたIDの商品を更新します。
    """
    if item_id not in db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Item not found"
        )
    
    stored_item = db[item_id]
    
    # ItemUpdateモデルから更新データ（Noneでない値のみ）を取得
    update_data = item_update.model_dump(exclude_unset=True)
    
    # stored_itemのコピーを更新データで上書き
    updated_item = stored_item.model_copy(update=update_data)
    
    db[item_id] = updated_item
    return updated_item
```

**解説:**

*   `@app.put(...)`: HTTP PUTリクエストを受け付けます。部分的な更新には `PATCH` が使われることも多いですが、ここでは `PUT` を例とします。
*   `item_update.model_dump(exclude_unset=True)`: `ItemUpdate` モデルから辞書を作成する際に、クライアントから送信されなかった（デフォルト値のままの）フィールドを除外します。これにより、送信されたフィールドだけを更新できます。
*   `stored_item.model_copy(update=update_data)`: Pydantic v2で推奨される方法です。元のモデルのコピーを作成し、`update` 引数で指定されたデータでフィールドを上書きします。

##### **Delete: 商品の削除**

`main.py` に以下を追加します。

```python
@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    """
    指定されたIDの商品を削除します。
    """
    if item_id not in db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Item not found"
        )
    del db[item_id]
    return
```

**解説:**

*   `@app.delete(...)`: HTTP DELETEリクエストを受け付けます。
*   `status_code=status.HTTP_204_NO_CONTENT`: 処理成功時にコンテンツを返さないことを示すステータスコード204を返します。この場合、FastAPIはレスポンスボディを送信しません。そのため、`response_model` は不要で、関数の戻り値もありません。

#### 1.4. 実行と確認

以下のコマンドでAPIサーバーを起動します。

```bash
uvicorn main:app --reload
```

ブラウザで `http://127.0.0.1:8000/docs` を開くと、Swagger UIによる自動対話ドキュメントが表示されます。ここで各エンドポイントの動作を直接テストできます。

---

### 2. データベース連携 (SQLAlchemy)

実務アプリケーションでは、データを永続化するためにデータベースが必須です。ここでは、Pythonで広く使われているORM（Object-Relational Mapper）であるSQLAlchemyと、PostgreSQLを非同期で扱うための `asyncpg` ドライバーを使って、FastAPIアプリケーションをデータベースに接続します。

#### 2.1. 必要なライブラリのインストール

```bash
pip install sqlalchemy "asyncpg[sa]"
```
`asyncpg[sa]` とすることで、SQLAlchemyがasyncpgを非同期ドライバーとして認識するために必要なアダプターもインストールされます。

#### 2.2. ディレクトリ構成

プロジェクトが複雑になるのに備え、ファイルを機能ごとに分割します。

```
/my_project
├── alembic/              # DBマイグレーション用 (後述)
├── alembic.ini           # Alembic設定ファイル
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPIアプリケーションのエントリーポイント
│   ├── crud.py           # CRUD操作のロジック
│   ├── database.py       # データベース接続設定
│   ├── models.py         # SQLAlchemyモデル
│   └── schemas.py        # Pydanticモデル (スキーマ)
└── ...
```

#### 2.3. データベース接続設定 (`database.py`)

データベースへの接続設定と、セッションを管理するためのコードを記述します。

`app/database.py`
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# PostgreSQLの接続URL (環境変数から取得するのが望ましい)
# フォーマット: "postgresql+asyncpg://user:password@host/dbname"
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/fastapi_db"

# 非同期エンジンを作成
engine = create_async_engine(DATABASE_URL, echo=True)

# 非同期セッションを作成するためのSessionLocalクラス
# expire_on_commit=False にすることで、コミット後もオブジェクトにアクセスできる
AsyncSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine, 
    class_=AsyncSession,
    expire_on_commit=False
)

# SQLAlchemyモデルのベースクラス
Base = declarative_base()

# DI (依存性注入) のためのDBセッション取得関数
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

**解説:**

*   `create_async_engine`: SQLAlchemyで非同期操作を行うためのエンジンを作成します。`echo=True` は実行されるSQLクエリをコンソールに出力する設定で、デバッグに便利です。
*   `AsyncSessionLocal`: データベースとの対話を行うためのセッションを作成するファクトリです。
*   `Base`: これを継承して、後述するデータベースのテーブルに対応するモデルクラスを作成します。
*   `get_db()`: FastAPIの依存性注入システムで使われる関数です。リクエストごとに新しいDBセッションを開始し、処理が終わったら自動でセッションを閉じる役割を持ちます。`async with`構文により、セッションの開始と終了（ロールバックやクローズを含む）が確実に行われます。`yield` でセッションをエンドポイント関数に提供します。

#### 2.4. SQLAlchemyモデルの定義 (`models.py`)

データベースのテーブル構造をPythonクラスとして定義します。

`app/models.py`
```python
from sqlalchemy import Column, Integer, String, Float
from .database import Base

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    price = Column(Float, nullable=False)
    description = Column(String, nullable=True)
```

**解説:**

*   `__tablename__`: このモデルが対応するデータベースのテーブル名を指定します。
*   `Column`: テーブルのカラムを定義します。`primary_key=True` は主キー、`index=True` はインデックスを作成することを示し、検索パフォーマンスを向上させます。

#### 2.5. Pydanticスキーマの定義 (`schemas.py`)

APIの入出力に使われるPydanticモデルを定義します。前述の `main.py` にあったものをこちらに移動・整理します。

`app/schemas.py`
```python
from pydantic import BaseModel, Field
from typing import Optional

# --- ベースモデル ---
class ItemBase(BaseModel):
    name: str
    price: float = Field(..., gt=0)
    description: Optional[str] = None

# --- 作成用モデル ---
class ItemCreate(ItemBase):
    pass

# --- 読み取り用モデル (DBから取得したデータを返す) ---
class Item(ItemBase):
    id: int

    class Config:
        orm_mode = True # FastAPI 2.x / Pydantic v2では from_attributes = True
```
**Pydantic V2 Note:**
`orm_mode = True` は Pydantic V1 の書き方です。Pydantic V2 では `from_attributes = True` に変更されています。
```python
# Pydantic V2 の場合
class Item(ItemBase):
    id: int

    class Config:
        from_attributes = True
```

**解説:**

*   `ItemBase` を作成し、共通のフィールドをまとめることで、コードの重複を避けています。
*   `Config` クラスの `orm_mode` (または `from_attributes`) を `True` に設定することが重要です。これにより、Pydanticモデルは `item.name` のような属性アクセスでSQLAlchemyモデル（ORMオブジェクト）からデータを読み取れるようになります。

#### 2.6. CRUDロジックの実装 (`crud.py`)

データベース操作の具体的なロジックを、エンドポイントから分離してこのファイルにまとめます。これにより、ビジネスロジックが再利用しやすくなり、テストも容易になります。

`app/crud.py`
```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from . import models, schemas

# --- Read ---
async def get_item(db: AsyncSession, item_id: int):
    result = await db.execute(select(models.Item).filter(models.Item.id == item_id))
    return result.scalars().first()

async def get_items(db: AsyncSession, skip: int = 0, limit: int = 100):
    result = await db.execute(select(models.Item).offset(skip).limit(limit))
    return result.scalars().all()

# --- Create ---
async def create_item(db: AsyncSession, item: schemas.ItemCreate):
    db_item = models.Item(**item.model_dump())
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

# --- Update ---
async def update_item(db: AsyncSession, item_id: int, item_update: schemas.ItemUpdate):
    db_item = await get_item(db, item_id)
    if not db_item:
        return None
    
    update_data = item_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_item, key, value)
        
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

# --- Delete ---
async def delete_item(db: AsyncSession, item_id: int):
    db_item = await get_item(db, item_id)
    if not db_item:
        return None
    await db.delete(db_item)
    await db.commit()
    return db_item
```

**解説:**

*   全ての関数は `db: AsyncSession` を第一引数に取り、非同期でデータベース操作を行います。
*   `select(models.Item)`: SQLAlchemy 2.0 スタイルのクエリ構築方法です。
*   `db.execute(...)`: 非同期でクエリを実行します。
*   `result.scalars().first()`: 結果セットから最初の1つのオブジェクトを取得します。
*   `result.scalars().all()`: 全ての結果をオブジェクトのリストとして取得します。
*   `db.add(db_item)`: セッションに変更（この場合は新規追加）を記録します。
*   `await db.commit()`: セッション内の変更をデータベースに反映させます。
*   `await db.refresh(db_item)`: コミット後のデータを（自動採番されたIDなどを含め）データベースから再取得し、Pythonオブジェクトに反映させます。
*   `await db.delete(db_item)`: オブジェクトを削除対象としてセッションに記録します。

#### 2.7. メインアプリケーションの修正 (`main.py`)

最後に `main.py` を修正し、これまでに作成した各モジュールを統合します。

`app/main.py`
```python
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from . import crud, models, schemas
from .database import engine, get_db

# アプリケーション起動時にDBテーブルを作成（本番ではAlembic推奨）
# models.Base.metadata.create_all(bind=engine) # 非同期ではこの方法は使えない

app = FastAPI()

@app.post("/items/", response_model=schemas.Item, status_code=status.HTTP_21_CREATED)
async def create_item_endpoint(
    item: schemas.ItemCreate, db: AsyncSession = Depends(get_db)
):
    return await crud.create_item(db=db, item=item)

@app.get("/items/", response_model=List[schemas.Item])
async def read_items_endpoint(
    skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)
):
    items = await crud.get_items(db, skip=skip, limit=limit)
    return items

@app.get("/items/{item_id}", response_model=schemas.Item)
async def read_item_endpoint(item_id: int, db: AsyncSession = Depends(get_db)):
    db_item = await crud.get_item(db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.put("/items/{item_id}", response_model=schemas.Item)
async def update_item_endpoint(
    item_id: int, item: schemas.ItemUpdate, db: AsyncSession = Depends(get_db)
):
    db_item = await crud.update_item(db, item_id=item_id, item_update=item)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.delete("/items/{item_id}", response_model=schemas.Item)
async def delete_item_endpoint(item_id: int, db: AsyncSession = Depends(get_db)):
    db_item = await crud.delete_item(db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

```

**解説:**

*   `Depends(get_db)`: これがFastAPIの強力な依存性注入システムです。各エンドポイント関数が呼び出されるたびに、FastAPIは `get_db` 関数を実行し、その `yield` した値（この場合は `AsyncSession`）を `db` 引数に渡します。リクエスト処理が終わると、`get_db` 関数の `yield` 以降の処理（セッションのクローズ）が実行されます。これにより、エンドポイントのコードはDBセッションの管理を意識する必要がなくなります。

#### 2.8. データベースマイグレーション (Alembic)

`Base.metadata.create_all(engine)` は開発初期には便利ですが、本番環境では既存のデータを破壊する可能性があるため使用すべきではありません。代わりに、データベーススキーマのバージョン管理ツールである **Alembic** を使用します。

1.  **インストール**:
    ```bash
    pip install alembic
    ```

2.  **初期化**: プロジェクトのルートディレクトリでコマンドを実行します。
    ```bash
    alembic init alembic
    ```
    これにより `alembic` ディレクトリと `alembic.ini` 設定ファイルが生成されます。

3.  **設定 (`alembic.ini`)**: `sqlalchemy.url` を `database.py` で定義した `DATABASE_URL` に設定します。
    ```ini
    sqlalchemy.url = postgresql+asyncpg://user:password@localhost/fastapi_db
    ```

4.  **設定 (`alembic/env.py`)**: Alembicがモデル定義を認識できるように設定します。
    ```python
    # ...
    from app.models import Base  # app/models.pyのBaseをインポート
    # ...
    target_metadata = Base.metadata # この行を修正
    # ...
    ```

5.  **マイグレーションファイルの生成**: モデルに変更を加えた後、以下のコマンドで差分を検出してマイグレーションスクリプトを自動生成します。
    ```bash
    alembic revision --autogenerate -m "Create items table"
    ```

6.  **データベースへの適用**: 生成されたスクリプトを使って、データベースのスキーマを更新します。
    ```bash
    alembic upgrade head
    ```

Alembicを導入することで、モデルの変更履歴をコードとして管理し、安全にデータベーススキーマを更新できるようになります。

---

### 3. 認証と認可 (OAuth2とJWT)

多くのAPIでは、特定のユーザーのみがリソースにアクセスできるように、認証（誰であるかを確認する）と認可（何をする権限があるかを確認する）の仕組みが必要です。ここでは、モダンなWebアプリケーションで標準的に使われる **OAuth2** と **JWT (JSON Web Token)** を組み合わせた認証を実装します。

具体的には、「パスワードフロー」（Resource Owner Password Credentials Grant）を実装し、ユーザーがユーザー名とパスワードを送信してアクセストークン（JWT）を取得し、その後のリクエストではそのトークンを使って認証を行います。

#### 3.1. 必要なライブラリのインストール

```bash
# JWTの操作とパスワードのハッシュ化
pip install "python-jose[cryptography]" "passlib[bcrypt]"

# Pythonの標準ライブラリにはない、OAuth2フローで必要なフォームデータを扱う
pip install python-multipart```

#### 3.2. ユーザーモデルの追加

`app/models.py` に `User` モデルを追加し、`app/schemas.py` に関連するスキーマを追加します。

`app/models.py`
```python
# ... (Itemモデルの下に追加)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
```

`app/schemas.py`
```python
# ... (Itemスキーマの下に追加)

# --- User ---
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True
```
パスワードはレスポンスに含めないため、`User` スキーマには `password` フィールドがありません。

#### 3.3. CRUD操作の追加 (`crud.py`)

ユーザーを作成し、ユーザー名で検索するためのCRUD関数を追加します。

`app/crud.py`
```python
# ... (import文に追記)
from passlib.context import CryptContext

# ... (itemのCRUD関数の下に追加)

# パスワードハッシュ化の設定
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- User CRUD ---
async def get_user_by_username(db: AsyncSession, username: str):
    result = await db.execute(select(models.User).filter(models.User.username == username))
    return result.scalars().first()

async def create_user(db: AsyncSession, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
```

**解説:**

*   `passlib.context.CryptContext`: パスワードのハッシュ化と検証を安全に行うためのライブラリです。`bcrypt` アルゴリズムを使用しています。
*   `get_password_hash`: 生のパスワードを受け取り、ハッシュ化して返します。データベースにはこのハッシュ化されたパスワードを保存します。
*   `verify_password`: ユーザーが入力したパスワードと、データベースに保存されているハッシュ化されたパスワードを比較し、一致するかどうかを検証します。

#### 3.4. 認証ロジックの実装 (`auth.py`)

認証関連のロジック（JWTの作成・検証、現在のユーザーを取得する依存関係など）をまとめる `auth.py` を新規に作成します。

`app/auth.py`
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from . import crud, models, schemas
from .database import get_db

# --- 設定 ---
# これらは環境変数など外部から読み込むべき
SECRET_KEY = "YOUR_SECRET_KEY" # 秘密鍵 (十分にランダムな文字列にする)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2のパスワードフローを定義
# tokenUrlはトークンを取得するためのエンドポイントのパス
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- パスワード関連 (crud.pyから移動または再利用) ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- JWT関連 ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- 依存関係: 現在のユーザーを取得 ---
async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.UserBase(username=username) # 簡単なバリデーション
    except JWTError:
        raise credentials_exception
    
    user = await crud.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```

**解説:**

*   `SECRET_KEY`: JWTの署名に使う秘密鍵です。これは絶対に外部に漏らしてはいけません。実務では環境変数やシークレット管理サービスから読み込みます。
*   `OAuth2PasswordBearer`: FastAPIのセキュリティユーティリティです。`Authorization: Bearer <token>` ヘッダーからトークンを自動で抽出する依存関係を提供します。`tokenUrl` には、トークンを発行するエンドポイントのURLを指定します。
*   `create_access_token`: ユーザー情報（`sub`（subject）としてユーザー名を入れるのが一般的）と有効期限（`exp`）を含むペイロードを作成し、`SECRET_KEY` で署名してJWTを生成します。
*   `get_current_user`: これが認証の核となる依存関係です。
    1.  `Depends(oauth2_scheme)` によってリクエストヘッダーからBearerトークンを取得します。
    2.  `jwt.decode` でトークンを検証・デコードします。有効期限切れや署名が不正な場合は `JWTError` が発生します。
    3.  デコードしたペイロードからユーザー名を取得し、データベースから該当ユーザーを検索します。
    4.  ユーザーが見つかればそのユーザーオブジェクトを返し、見つからなければ認証エラーを返します。
*   `get_current_active_user`: `get_current_user` に依存し、さらにそのユーザーがアクティブ（`is_active` が `True`）であるかを確認する依存関係です。保護したいエンドポイントではこちらを使うのがより安全です。

#### 3.5. 認証用エンドポイントの追加 (`main.py`)

ユーザー登録とトークン発行のためのエンドポイントを `main.py` に追加します。

`app/main.py`
```python
# ... (import文に追記)
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from . import auth

# --- トークン発行用スキーマ ---
class Token(BaseModel):
    access_token: str
    token_type: str

# ... (FastAPIインスタンスの下)

@app.post("/users/", response_model=schemas.User)
async def create_user_endpoint(
    user: schemas.UserCreate, db: AsyncSession = Depends(get_db)
):
    db_user = await crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return await crud.create_user(db=db, user=user)

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
):
    user = await crud.get_user_by_username(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=schemas.User)
async def read_users_me(
    current_user: models.User = Depends(auth.get_current_active_user)
):
    return current_user
```

**解説:**

*   `/users/` (POST): 新しいユーザーを登録するエンドポイントです。
*   `/token` (POST): トークンを発行するエンドポイントです。
    *   `OAuth2PasswordRequestForm = Depends()`: `username` と `password` を含むフォームデータ (`application/x-www-form-urlencoded`) を受け取るための特殊な依存関係です。
    *   ユーザーを検索し、パスワードを検証します。
    *   成功すれば、`create_access_token` を呼び出してJWTを生成し、クライアントに返します。
*   `/users/me/`: 認証が必要なエンドポイントの例です。
    *   `Depends(auth.get_current_active_user)` を使うことで、このエンドポイントは有効なJWTを持つリクエストしか受け付けなくなります。
    *   認証に成功した場合、`current_user` 引数に認証されたユーザーの `User` モデルオブジェクトが渡されるため、関数内でそのユーザー情報を利用できます。

#### 3.6. 保護されたエンドポイントの作成

既存の商品作成エンドポイントを、認証されたユーザーのみが実行できるように変更してみましょう。

`app/main.py`
```python
# create_item_endpointを修正
@app.post("/items/", response_model=schemas.Item, status_code=status.HTTP_201_CREATED)
async def create_item_endpoint(
    item: schemas.ItemCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user) # この行を追加
):
    # current_user を使ったロジックを追加することも可能 (例: 誰が作成したかを記録するなど)
    return await crud.create_item(db=db, item=item)
```

**解説:**

*   `Depends(auth.get_current_active_user)` を追加するだけで、このエンドポイントは認証を要求するようになります。リクエストに有効な `Authorization: Bearer <token>` ヘッダーが含まれていない場合、自動的に401 Unauthorizedエラーが返されます。
*   Swagger UI (`/docs`) もこの変更を自動で認識し、右上に「Authorize」ボタンが表示されるようになります。ここでトークンを入力すると、ドキュメント上から保護されたAPIを試すことができます。

---

### 4. 高度な機能と実践的なテクニック

ここでは、大規模アプリケーションの開発や運用に役立つ、より高度なFastAPIの機能を見ていきます。

#### 4.1. ルーター (`APIRouter`)

アプリケーションが大きくなると、すべてのエンドポイントを `main.py` に記述するのは見通しが悪くなります。`APIRouter` を使うと、関連するエンドポイントを機能ごとにファイルに分割できます。

1.  **ルーターファイルの作成 (`routers/items.py`)**:
    ```
    /my_project
    ├── app/
    │   ├── routers/
    │   │   ├── __init__.py
    │   │   └── items.py      # 商品関連のルーター
    │   ├── __init__.py
    │   ├── main.py
    │   └── ...
    ```

    `app/routers/items.py`
    ```python
    from fastapi import APIRouter, Depends, HTTPException, status
    from sqlalchemy.ext.asyncio import AsyncSession
    from typing import List

    from .. import crud, models, schemas, auth
    from ..database import get_db

    router = APIRouter(
        prefix="/items",
        tags=["items"],
        dependencies=[Depends(auth.get_current_active_user)], # このルーター全体に認証を適用
        responses={404: {"description": "Not found"}},
    )

    # @app.post(...) だった部分を @router.post(...) に変更
    # パスは prefix を除いた部分 ("/items/" -> "/")
    @router.post("/", response_model=schemas.Item, status_code=status.HTTP_201_CREATED)
    async def create_item(
        item: schemas.ItemCreate, 
        db: AsyncSession = Depends(get_db),
        # current_user はルーターの dependencies で取得される
        current_user: models.User = Depends(auth.get_current_active_user)
    ):
        return await crud.create_item(db=db, item=item)

    @router.get("/", response_model=List[schemas.Item])
    async def read_items(
        skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)
    ):
        return await crud.get_items(db, skip=skip, limit=limit)
    
    # ... 他の /items エンドポイントも同様にここに移動する
    ```

    **解説:**
    *   `APIRouter` のインスタンスを作成します。
    *   `prefix="/items"`: このルーター内のすべてのパスの先頭に `/items` が付きます。
    *   `tags=["items"]`: APIドキュメントでグループ化するためのタグです。
    *   `dependencies=[...]`: このルーター内のすべてのエンドポイントに適用される依存関係を指定できます。ここでは認証を共通で適用しています。

2.  **メインアプリケーションへの登録 (`main.py`)**:
    `main.py` から `items` 関連のエンドポイントを削除し、代わりにルーターをインクルードします。

    `app/main.py`
    ```python
    from fastapi import FastAPI
    from .routers import items # ルーターをインポート
    # ... 他のimport

    app = FastAPI()

    # ルーターをアプリケーションに含める
    app.include_router(items.router)

    # ユーザー登録やトークン発行など、/items 以外のエンドポイントはここに残す
    # ...
    ```

#### 4.2. バックグラウンドタスク (`BackgroundTasks`)

メール送信や重いデータ処理など、レスポンスを返すのを待つ必要がない処理は、バックグラウンドで実行するのが望ましいです。

```python
from fastapi import BackgroundTasks, Depends

# 例: ユーザー登録後にウェルカムメールを送信する
def send_welcome_email(email: str, username: str):
    # ここに実際のメール送信処理を書く (時間はかかるものと仮定)
    print(f"Sending welcome email to {email} for user {username}")
    # import time; time.sleep(5) 
    print("Email sent.")

@app.post("/users/register")
async def register_user(
    user_data: schemas.UserCreate,
    background_tasks: BackgroundTasks
):
    # ユーザー作成処理 ...
    new_user = ...

    # バックグラウンドタスクを追加
    background_tasks.add_task(send_welcome_email, new_user.email, new_user.username)
    
    return {"message": "User registered. Welcome email is being sent in the background."}
```
**解説:**

*   エンドポイントの引数に `background_tasks: BackgroundTasks` を追加します。
*   `background_tasks.add_task(関数, 引数1, 引数2, ...)` を呼び出すと、FastAPIはレスポンスをクライアントに返した **後で**、指定された関数を非同期で実行します。これにより、クライアントの待ち時間を短縮できます。

#### 4.3. ミドルウェア (`Middleware`)

ミドルウェアは、すべてのリクエストがエンドポイントに到達する前、およびすべてのレスポンスがクライアントに返される前に実行される処理です。リクエストのロギング、CORSの設定、カスタムヘッダーの追加などに使用されます。

```python
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- CORSミドルウェア ---
# 異なるオリジン(ドメイン)からのリクエストを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com", "http://localhost:3000"], # 許可するオリジンのリスト
    allow_credentials=True,
    allow_methods=["*"], # 全てのメソッドを許可
    allow_headers=["*"], # 全てのヘッダーを許可
)

# --- カスタムミドルウェア (リクエスト処理時間を計測) ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

```

**解説:**

*   `app.add_middleware(...)`: アプリケーションにミドルウェアを追加します。`CORSMiddleware` はFastAPIに組み込まれており、非常に便利です。
*   `@app.middleware("http")`: 独自のミドルウェアを作成するためのデコレータです。関数は `request: Request` と `call_next` の2つの引数を取ります。`await call_next(request)` が実際のエンドポイント処理を呼び出す部分です。その前後で共通の処理を追加できます。

#### 4.4. 設定管理 (`Pydantic Settings`)

`SECRET_KEY` や `DATABASE_URL` のような設定値をコードに直接書き込むのは避けるべきです。Pydanticの `Settings` を使うと、環境変数や `.env` ファイルから設定をスマートに読み込めます。

1.  **インストール**:
    ```bash
    pip install "pydantic-settings"
    ```

2.  **設定ファイルの作成 (`.env`)**: プロジェクトルートに作成します。
    ```
    DATABASE_URL="postgresql+asyncpg://user:password@localhost/fastapi_db"
    SECRET_KEY="a_very_long_and_random_secret_key_for_jwt"
    ACCESS_TOKEN_EXPIRE_MINUTES=30
    ```

3.  **設定管理クラスの作成 (`config.py`)**:
    `app/config.py`
    ```python
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        database_url: str
        secret_key: str
        access_token_expire_minutes: int = 30

        class Config:
            env_file = ".env"

    settings = Settings()
    ```

4.  **コード内での使用**:
    ハードコードされた値を、この `settings` オブジェクトからの読み込みに置き換えます。

    `app/database.py`
    ```python
    from .config import settings
    # DATABASE_URL = "..." を削除
    engine = create_async_engine(settings.database_url, echo=True)
    ```

    `app/auth.py`
    ```python
    from .config import settings
    # SECRET_KEY = "..." などを削除
    SECRET_KEY = settings.secret_key
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
    ```

**解説:**

*   `BaseSettings` を継承したクラスを定義すると、Pydanticは自動的に同名の環境変数を探し、フィールドに値を読み込みます。
*   `class Config` の `env_file` で `.env` ファイルのパスを指定できます。
*   これにより、設定とコードが分離され、開発環境と本番環境で異なる設定を容易に適用できるようになります。

---

### 5. テスト (`TestClient`)

FastAPIは、Pytestと組み合わせて簡単にテストが書けるように `TestClient` を提供しています。

1.  **インストール**:
    ```bash
    pip install pytest httpx
    ```

2.  **テスト用データベース設定**: テスト実行時には、本番用とは別のテスト用データベースを使うのが定石です。

    `app/tests/conftest.py` (Pytestのフィクスチャを定義するファイル)
    ```python
    import pytest
    from httpx import AsyncClient
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.main import app
    from app.database import Base, get_db

    TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db" # テスト用のSQLite DB

    engine = create_async_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

    # テスト用のDBセッションで get_db をオーバーライドする
    async def override_get_db():
        async with TestingSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    @pytest.fixture()
    async def client():
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # テストの前にテーブルを全作成
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            yield ac
            
            # テストの後にテーブルを全削除
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
    ```

3.  **テストコードの作成 (`tests/test_items.py`)**:
    `app/tests/test_items.py`
    ```python
    import pytest

    @pytest.mark.asyncio
    async def test_create_item(client):
        response = await client.post(
            "/items/", 
            json={"name": "Test Item", "price": 10.5, "description": "A test item"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Item"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_read_item(client):
        # 先にアイテムを作成
        response_create = await client.post(
            "/items/", json={"name": "Another Test Item", "price": 99.9}
        )
        item_id = response_create.json()["id"]

        # 作成したアイテムを読み取り
        response_read = await client.get(f"/items/{item_id}")
        assert response_read.status_code == 200
        data = response_read.json()
        assert data["name"] == "Another Test Item"
        assert data["id"] == item_id
    ```

**解説:**

*   `TestClient` (非同期版では `httpx.AsyncClient`) は、実行中のサーバーを必要とせず、直接FastAPIアプリケーションにリクエストを送信できるクライアントです。
*   `app.dependency_overrides`: FastAPIの強力な機能で、テスト時に特定の依存関係（`get_db`など）を別のもの（テストDB用のセッションを返す関数）に差し替えることができます。
*   Pytestのフィクスチャ (`@pytest.fixture`) を使うことで、テストのセットアップ（DBテーブル作成）と後片付け（DBテーブル削除）を共通化できます。
*   テストコードは、実際のHTTPリクエストと同様に `client.post`, `client.get` などを使い、レスポンスのステータスコードやJSONボディを `assert` で検証します。
