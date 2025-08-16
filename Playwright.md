

### 1. Playwrightとは？ その魅力と特徴

Playwrightがなぜこれほどまでに注目されているのか、その核心となる特徴から見ていきましょう。

#### 1.1. Playwrightの主な特徴

*   **クロスブラウザ対応**: 1つのコードでChromium、Firefox、WebKitを全て自動化できます。これにより、ブラウザ間の互換性テストが非常に簡単になります。
*   **高速かつ信頼性の高い実行**: Playwrightには**Auto-Wait（自動待機）**機能が組み込まれています。これは、要素が表示されたり、クリック可能になったりするまでPlaywrightが自動で待ってくれる機能です。これにより、「要素が見つからない」といったタイミングに起因する不安定なテストが劇的に減少します。
*   **豊富な機能とツール**:
    *   **Codegen（コード生成）**: ブラウザ上で行った操作を記録し、テストコードを自動で生成します。
    *   **Playwright Inspector**: ステップ実行しながらテストをデバッグできるツールです。
    *   **Trace Viewer**: テスト実行の全記録（スクリーンショット、DOMスナップショット、ネットワークリクエストなど）を視覚的に確認でき、失敗原因の特定が非常に容易になります。
*   **複数言語対応**: TypeScript、JavaScript、Python、Java、.NET (C#)に対応しており、プロジェクトで使われている言語に合わせて導入できます。
*   **高度な自動化機能**: ファイルのアップロード/ダウンロード、認証状態の保存、ネットワークリクエストの傍受や変更、APIテストなど、Webアプリケーションのテストに必要な機能が網羅されています。

#### 1.2. 他のツールとの比較

| 特徴 | Playwright | Selenium | Cypress |
| :--- | :--- | :--- | :--- |
| **開発元** | Microsoft | オープンソースコミュニティ | Cypress.io |
| **アーキテクチャ** | WebDriverプロトコル不使用 | WebDriverプロトコル使用 | ブラウザ内で直接実行 |
| **クロスブラウザ** | ◎ (Chromium, Firefox, WebKit) | ◎ (主要ブラウザに対応) | △ (WebKitのサポートは実験的) |
| **自動待機** | ◎ (標準装備) | △ (明示的な待機処理が必要) | ◎ (標準装備) |
| **テストランナー** | ◎ (Playwright Testが同梱) | ✕ (別途用意が必要) | ◎ (専用ランナーが同梱) |
| **ネットワーク操作** | ◎ (リクエストの傍受/変更が可能) | △ (限定的) | ◎ (リクエストの傍受/変更が可能) |
| **Trace Viewer** | ◎ (非常に強力なデバッグツール) | ✕ | ✕ (タイムトラベルデバッグ機能あり) |

Playwrightは、Seleniumの広範なブラウザ・言語対応と、Cypressのようなモダンな開発体験やデバッグのしやすさを両立させた、まさに「いいとこ取り」のフレームワークと言えます。

### 2. 環境構築

Playwrightを始めるための環境構築は非常に簡単です。ここでは、Node.js環境でのセットアップを解説します。

#### 2.1. Node.jsのインストール

まず、お使いのコンピュータにNode.jsがインストールされている必要があります。インストールされていない場合は、公式サイトからLTS（長期サポート）版をインストールしてください。

#### 2.2. Playwrightのインストール

1.  プロジェクト用のディレクトリを作成し、そのディレクトリに移動します。

    ```bash
    mkdir playwright-tutorial
    cd playwright-tutorial
    ```

2.  以下のコマンドを実行して、Playwrightを初期化します。

    ```bash
    npm init playwright@latest
    ```

3.  コマンドを実行すると、対話形式でいくつか質問されます。基本的にはデフォルトのままでEnterキーを押していけば問題ありません。
    *   **Use TypeScript or JavaScript?** (TypeScriptが推奨されます)
    *   **Name of your tests folder?** (デフォルトは `tests`)
    *   **Add a GitHub Actions workflow?** (CI/CDを考えているなら `true`)
    *   **Install Playwright browsers?** (`true` を選択)

4.  インストールが完了すると、以下のファイルとディレクトリが生成されます。
    *   `playwright.config.ts`: Playwrightの設定ファイルです。
    *   `tests/`: テストコードを格納するディレクトリです。
    *   `tests-examples/`: サンプルコードが格納されています。参考にすると良いでしょう。
    *   `package.json`: プロジェクトの依存関係などが記述されています。
    *   `node_modules/`: インストールされたパッケージが格納されます。

これで、Playwrightを実行する準備が整いました。

### 3. Playwrightの基本的な使い方（Core Concepts）

ここでは、Playwrightを操作する上での中心的な概念である「ロケーター」「アクション」「アサーション」について学びます。

#### 3.1. テストファイルの基本構造

`tests/` ディレクトリに新しいファイル `example.spec.ts` を作成しましょう。Playwright Testは、`test`関数と`expect`関数を使ってテストを記述します。

```typescript
// tests/example.spec.ts

import { test, expect } from '@playwright/test';

// test関数でテストを定義します。
// 第一引数: テスト名
// 第二引数: テスト処理を記述するコールバック関数
test('基本的な操作とアサーションの例', async ({ page }) => {
  // page: ブラウザのページを操作するためのオブジェクト。Playwrightが自動的に用意してくれます。

  // 1. 特定のURLにアクセス
  await page.goto('https://playwright.dev/');

  // 2. ページタイトルを検証
  await expect(page).toHaveTitle(/Playwright/);

  // 3. 要素を特定してクリック
  const getStartedLink = page.getByRole('link', { name: 'Get started' });
  await getStartedLink.click();

  // 4. 新しいページのURLを検証
  await expect(page).toHaveURL(/.*intro/);
});
```

#### 3.2. 要素の特定（ロケーター）

Webページを自動操作するには、まず操作対象の要素（ボタン、入力欄など）を正確に特定する必要があります。Playwrightでは、この要素を特定するための仕組みを**ロケーター (Locator)** と呼びます。

Playwrightでは、ユーザーがどのようにページを認識するかに基づいた、**人間中心のロケーター**が推奨されています。

*   `page.getByRole(role, options)`: **最も推奨されるロケーター**です。`button`, `link`, `heading` といったWAI-ARIAロールに基づいて要素を探します。スクリーンリーダーなど、支援技術のユーザーがページをどのようにナビゲートするかに近いため、非常に堅牢です。
    ```typescript
    // "Sign in" という名前のボタンを探す
    await page.getByRole('button', { name: 'Sign in' }).click();

    // "Docs" という名前のリンクを探す
    await page.getByRole('link', { name: 'Docs' }).click();

    // レベル1の見出し (<h1>) を探す
    const heading = page.getByRole('heading', { level: 1 });
    ```

*   `page.getByText(text)`: 表示されているテキストで要素を探します。
    ```typescript
    // "Welcome" というテキストを持つ要素をクリック
    await page.getByText('Welcome').click();
    ```

*   `page.getByLabel(text)`: `<label>` 要素のテキストに関連付けられたフォームの入力要素を探します。
    ```typescript
    // "Password" というラベルを持つ入力欄にテキストを入力
    await page.getByLabel('Password').fill('s3cr3tP@ssw0rd');
    ```

*   `page.getByPlaceholder(text)`: `placeholder` 属性の値で入力欄を探します。
    ```typescript
    // "name@example.com" というプレースホルダーを持つ入力欄を探す
    await page.getByPlaceholder('name@example.com').fill('test@example.com');
    ```

*   `page.getByTestId(testId)`: `data-testid` 属性を使って要素を探します。これは、テストのためだけにHTMLに付与するIDで、開発者と協力して埋め込むことで、非常に安定したテストを書くことができます。
    ```html
    <!-- HTML側の記述 -->
    <button data-testid="submit-button">Submit</button>
    ```
    ```typescript
    // テストコード側の記述
    await page.getByTestId('submit-button').click();
    ```

CSSセレクタ (`page.locator('.my-class')`) やXPath (`page.locator('//button')`) も利用可能ですが、これらは実装の変更に弱いため、上記の方法で特定できない場合の最終手段として考えるのが良いでしょう。

#### 3.3. 要素の操作（アクション）

ロケーターで要素を特定したら、次はその要素に対して操作を行います。

*   `locator.click()`: 要素をクリックします。
    ```typescript
    await page.getByRole('button', { name: 'ログイン' }).click();
    ```

*   `locator.fill(value)`: 入力欄のテキストをクリアしてから、新しいテキストを入力します。
    ```typescript
    await page.getByLabel('ユーザー名').fill('my-username');
    ```

*   `locator.press(key)`: キーボードのキーを押します。
    ```typescript
    // Enterキーを押す
    await page.getByLabel('検索').press('Enter');
    ```

*   `locator.check()` / `locator.uncheck()`: チェックボックスをオン/オフします。
    ```typescript
    await page.getByLabel('同意する').check();
    ```

*   `locator.selectOption(value)`: ドロップダウンリストからオプションを選択します。
    ```typescript
    await page.getByLabel('国を選択').selectOption('Japan');
    ```

*   `locator.hover()`: 要素にマウスカーソルを合わせます。
    ```typescript
    await page.getByRole('button', { name: 'メニュー' }).hover();
    ```

*   `locator.screenshot()`: 特定の要素だけのスクリーンショットを撮ります。
    ```typescript
    await page.getByRole('heading', { name: 'ようこそ' }).screenshot({ path: 'heading.png' });
    ```

#### 3.4. 検証（アサーション）

操作の結果が期待通りであるかを確認するのがアサーションです。Playwright Testでは `expect` を使います。Playwrightのアサーションは**Web-First Assertions**と呼ばれ、自動待機機能が組み込まれています。

例えば、`expect(locator).toBeVisible()` は、要素がすぐに見えなくても、タイムアウトするまで表示されるのを待ち続けます。これにより、テストの信頼性が向上します。

*   `expect(locator).toBeVisible()`: 要素が画面に表示されていることを検証します。
    ```typescript
    await expect(page.getByText('ログイン成功')).toBeVisible();
    ```

*   `expect(locator).toHaveText(text)`: 要素が特定のテキストを含むことを検証します。
    ```typescript
    const errorMessage = page.locator('.error-message');
    await expect(errorMessage).toHaveText('パスワードが違います');
    ```

*   `expect(locator).toBeEnabled()`: 要素が操作可能（有効）であることを検証します。
    ```typescript
    // フォーム入力が完了するまでSubmitボタンは無効になっている、というシナリオで利用
    const submitButton = page.getByRole('button', { name: '送信' });
    await expect(submitButton).toBeEnabled();
    ```

*   `expect(page).toHaveURL(url)`: 現在のページのURLを検証します。
    ```typescript
    await expect(page).toHaveURL('https://example.com/dashboard');
    ```

*   `expect(page).toHaveTitle(title)`: 現在のページのタイトルを検証します。
    ```typescript
    await expect(page).toHaveTitle('ダッシュボード');
    ```

### 4. Playwright Test（テストランナー）

Playwrightには、`@playwright/test` という強力なテストランナーが付属しています。これにより、テストの実行、管理、デバッグが非常に簡単になります。

#### 4.1. テストの実行

ターミナルで以下のコマンドを実行するだけで、`tests` ディレクトリ内のすべてのテストが実行されます。

```bash
npx playwright test
```

デフォルトでは、`playwright.config.ts` で設定されたすべてのブラウザ（Chromium, Firefox, WebKit）で、**ヘッドレスモード**（画面表示なし）で実行されます。

**便利な実行オプション:**

*   **特定のファイルを実行**:
    ```bash
    npx playwright test tests/my-test.spec.ts
    ```
*   **特定のブラウザで実行**:
    ```bash
    npx playwright test --project=chromium
    ```
*   **UIモードで実行**:
    ```bash
    npx playwright test --ui
    ```
    UIモードは、テストの実行状況をグラフィカルなインターフェースで確認できる非常に便利な機能です。各ステップでのDOMの状態を確認したり、時間を巻き戻してデバッグ（タイムトラベルデバッグ）したりできます。
*   **デバッグモードで実行**:
    ```bash
    npx playwright test --debug
    ```
    Playwright Inspectorが起動し、ステップ実行やロケーターの確認ができます。

#### 4.2. Hooks とテストのグループ化

テストには、繰り返し行われる前処理や後処理があります。これらを効率的に記述するためにHooksが用意されています。

*   `test.beforeEach(callback)`: 各テストの前に実行されます。ログイン処理やテストデータの準備などによく使われます。
*   `test.afterEach(callback)`: 各テストの後に実行されます。ログアウトやテストデータのクリーンアップなどに使われます。
*   `test.beforeAll(callback)`: ファイル内の最初のテストの前に一度だけ実行されます。
*   `test.afterAll(callback)`: ファイル内の最後のテストの後に一度だけ実行されます。

また、関連するテストは `test.describe` を使ってグループ化できます。

```typescript
// tests/login.spec.ts
import { test, expect } from '@playwright/test';

test.describe('ログイン機能のテスト', () => {
  // このdescribeブロック内のすべてのテストの前に、ログインページにアクセスする
  test.beforeEach(async ({ page }) => {
    await page.goto('https://example.com/login');
  });

  test('正しい認証情報でログインできる', async ({ page }) => {
    await page.getByLabel('ユーザーID').fill('user');
    await page.getByLabel('パスワード').fill('password');
    await page.getByRole('button', { name: 'ログイン' }).click();

    await expect(page.getByRole('heading', { name: 'ようこそ' })).toBeVisible();
  });

  test('間違ったパスワードではログインできない', async ({ page }) => {
    await page.getByLabel('ユーザーID').fill('user');
    await page.getByLabel('パスワード').fill('wrong-password');
    await page.getByRole('button', { name: 'ログイン' }).click();

    await expect(page.getByText('IDまたはパスワードが違います')).toBeVisible();
  });
});
```

### 5. 高度な機能

Playwrightの真価は、テスト開発を劇的に効率化する高度なツール群にあります。

#### 5.1. Trace Viewer: 最強のデバッグツール

テストが失敗した際、なぜ失敗したのかを特定するのは非常に時間がかかる作業です。**Trace Viewer** はこの問題を解決します。

**Traceの有効化:**

`playwright.config.ts` で `trace` オプションを設定します。テストが初めて失敗した時にトレースファイルを保存する `on-first-retry` がおすすめです。

```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  // ...
  trace: 'on-first-retry',
});
```

**Traceの確認:**

テスト実行後、`test-results` ディレクトリに `trace.zip` というファイルが生成されます。以下のコマンドでTrace Viewerを起動します。

```bash
npx playwright show-trace test-results/your-test-trace.zip
```

ブラウザで以下のような画面が開き、テストの全容を可視化できます。

*   **アクションのタイムライン**: 各ステップが時系列で表示されます。
*   **DOMスナップショット**: 各アクション実行前後のDOMの状態を確認できます。
*   **スクリーンショット**: 各アクションの瞬間のスクリーンショットが表示されます。
*   **ネットワークログ**: テスト中に発生したすべてのネットワークリクエストとレスポンスを確認できます。
*   **コンソールログ**: ブラウザのコンソールに出力されたログを確認できます。

**「CIサーバーでは成功するのに、ローカルでは失敗する」といった厄介な問題も、Trace Viewerを使えば原因を瞬時に特定できます。**

#### 5.2. Codegen: 操作を記録してコードを生成

Codegenは、ブラウザでの手動操作をPlaywrightのテストコードに変換してくれるツールです。

以下のコマンドを実行すると、空のブラウザウィンドウとPlaywright Inspectorが起動します。

```bash
npx playwright codegen https://playwright.dev/
```

ブラウザ上で行ったクリックや入力などの操作が、リアルタイムでInspectorウィンドウにコードとして記録されていきます。これは、テストの雛形を作成したり、複雑な操作のセレクタを調べたりするのに非常に役立ちます。

#### 5.3. 認証の管理 (Authentication)

多くのテストでは、ログイン状態であることが前提となります。しかし、各テストの開始時に毎回ログイン処理を実行するのは非効率です。

Playwrightでは、一度ログインした後の**認証状態（CookieやlocalStorageなど）をファイルに保存し、他のテストで再利用する**ことができます。

**1. 認証状態を保存するセットアップファイルを作成**

`global.setup.ts`のようなファイル名で作成します。

```typescript
// global.setup.ts
import { test as setup, expect } from '@playwright/test';

const authFile = 'playwright/.auth/user.json';

setup('authenticate', async ({ page }) => {
  // ログイン処理を実行
  await page.goto('https://github.com/login');
  await page.getByLabel('Username or email address').fill('YOUR_USERNAME');
  await page.getByLabel('Password').fill('YOUR_PASSWORD');
  await page.getByRole('button', { name: 'Sign in' }).click();
  
  // ログインが成功したことを確認
  await expect(page.getByRole('button', { name: 'View profile and more' })).toBeVisible();

  // 現在のページの認証情報をファイルに保存
  await page.context().storageState({ path: authFile });
});
```

**2. `playwright.config.ts` でセットアップファイルを指定**

```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  // ...
  projects: [
    {
      name: 'setup',
      testMatch: /global\.setup\.ts/,
    },
    {
      name: 'chromium',
      use: {
        storageState: 'playwright/.auth/user.json', // 保存した認証情報を使用
      },
      dependencies: ['setup'], // 'setup'プロジェクトの完了後に実行
    },
    // ... firefox, webkit projects
  ],
});
```

**3. テストファイルではログイン済みの状態で開始**

これで、テストはログインページではなく、ログイン後のダッシュボードページから開始できます。

```typescript
// tests/dashboard.spec.ts
import { test, expect } from '@playwright/test';

test('ダッシュボードに自分のリポジトリが表示される', async ({ page }) => {
  // ログイン処理は不要！
  await page.goto('https://github.com/');
  
  // ログイン後のページ要素を直接操作・検証できる
  await expect(page.getByRole('link', { name: 'Your repositories' })).toBeVisible();
});
```

この方法は、テストの実行時間を大幅に短縮し、テストの関心を本来検証したい機能に集中させる上で非常に重要です。

#### 5.4. ネットワーク操作

Playwrightでは、ブラウザが送受信するネットワークリクエストを完全にコントロールできます。

*   **APIレスポンスのモック**:
    `page.route()` を使うと、特定のURLへのリクエストを傍受し、任意のレスポンスを返すことができます。これにより、バックエンドAPIが未完成でも、フロントエンドのテストを先行して進めることが可能になります。

    ```typescript
    test('TODOリストが空の場合の表示をテストする', async ({ page }) => {
      // /api/todosへのGETリクエストを傍受
      await page.route('**/api/todos', async route => {
        // 空の配列をJSONとして返す
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        });
      });

      await page.goto('/todos');
      
      // APIが空配列を返した結果、特定のメッセージが表示されることを確認
      await expect(page.getByText('タスクを追加してください')).toBeVisible();
    });
    ```

#### 5.5. APIテスト

PlaywrightはUIテストだけでなく、APIテストも実行できます。`request` fixture を使うことで、ブラウザを介さずに直接APIリクエストを送信できます。

```typescript
import { test, expect } from '@playwright/test';

test('POST /users で新しいユーザーを作成できる', async ({ request }) => {
  const response = await request.post('/api/users', {
    data: {
      name: 'Taro Yamada',
      email: 'taro@example.com',
    },
  });

  // レスポンスが成功したことを確認
  expect(response.ok()).toBeTruthy();

  const responseBody = await response.json();
  expect(responseBody).toHaveProperty('name', 'Taro Yamada');
});
```

UIテストの前処理としてAPI経由でテストデータを作成したり、UI操作の結果がAPIに正しく反映されているかを確認したりと、UIテストとAPIテストを組み合わせることで、より網羅的なテストが可能になります。

### 6. 実践的なテスト戦略

大規模なアプリケーションのテストを効率的に管理するためには、いくつかの設計パターンを導入することが推奨されます。

#### 6.1. Page Object Model (POM)

**Page Object Model (POM)** は、テストコードからロケーターや操作のロジックを分離し、ページごとにクラスとしてまとめる設計パターンです。

**メリット:**
*   **可読性の向上**: テストコードが「何をしているか」に集中でき、読みやすくなります。
*   **メンテナンス性の向上**: UIの変更があった場合、修正箇所がページオブジェクトクラスに限定されるため、メンテナンスが容易になります。
*   **再利用性の向上**: 複数のテストで同じページ操作を再利用できます。

**実装例:**

まず、ページオブジェクトクラスを作成します。

```typescript
// pages/LoginPage.ts
import type { Page, Locator } from '@playwright/test';

export class LoginPage {
  // Readonlyプロパティとしてページインスタンスとロケーターを定義
  readonly page: Page;
  readonly usernameInput: Locator;
  readonly passwordInput: Locator;
  readonly loginButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.usernameInput = page.getByLabel('ユーザーID');
    this.passwordInput = page.getByLabel('パスワード');
    this.loginButton = page.getByRole('button', { name: 'ログイン' });
  }

  // ページへの遷移もメソッドとして用意
  async goto() {
    await this.page.goto('https://example.com/login');
  }

  // ログイン操作を一つのメソッドにまとめる
  async login(username: string, password: string) {
    await this.usernameInput.fill(username);
    await this.passwordInput.fill(password);
    await this.loginButton.click();
  }
}
```

次に、このページオブジェクトをテストコードから利用します。

```typescript
// tests/login-with-pom.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';

test('POMを使ったログインテスト', async ({ page }) => {
  const loginPage = new LoginPage(page);

  await loginPage.goto();
  await loginPage.login('user', 'password');

  // 検証ロジックはテストコード側に記述
  await expect(page.getByRole('heading', { name: 'ようこそ' })).toBeVisible();
});
```

POMを使うことで、テストコードは `fill` や `click` といった具体的な操作から解放され、「ログインページに移動し、ログインする」という、より抽象的で分かりやすい記述になりました。

#### 6.2. CI/CDへの統合

Playwrightは、GitHub ActionsなどのCI/CDツールとの統合が非常に簡単です。

インストール時に `Add a GitHub Actions workflow?` で `true` を選択していれば、`.github/workflows/playwright.yml` というファイルが自動生成されます。

このワークフローは、コードがリポジトリにプッシュされるたびに自動で以下の処理を実行します。

1.  Playwrightの環境をセットアップ
2.  ブラウザをインストール
3.  テストを実行
4.  テスト結果をHTMLレポートとしてアーティファクトにアップロード

これにより、常にテストがパスしている状態を保ち、安心してデプロイを行うことができます。HTMLレポートはGitHub Actionsの実行結果ページからダウンロードでき、CI上で失敗したテストの詳細をTrace Viewerで確認することも可能です。

### 7. まとめ

この解説では、Playwrightの基本的な概念から、テスト開発を加速させる高度な機能、そして実践的なテスト戦略までを網羅的に見てきました。

**Playwrightの重要なポイント:**

*   **Auto-WaitとWeb-First Assertions**がテストの不安定さを解消し、信頼性を高めます。
*   **人間中心のロケーター** (`getByRole`など) を使うことで、堅牢でメンテナンスしやすいテストを記述できます。
*   **Trace Viewer** は、テストの失敗原因を特定するための非常に強力なデバッグツールです。
*   **Codegen** や **UIモード** は、テスト作成とデバッグのサイクルを高速化します。
*   **認証状態の保存**や**APIテスト**機能を活用することで、テストをより高速かつ効率的に実行できます。
*   **Page Object Model**を導入することで、大規模なプロジェクトでもテストコードのメンテナンス性を維持できます。

