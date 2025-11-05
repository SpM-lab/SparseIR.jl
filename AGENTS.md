# AGENTS.md - Julia Package (SparseIR.jl)

このファイルは、Juliaパッケージ`SparseIR.jl`で作業する際の詳細な指示を記載します。

## パッケージ概要

- **言語**: Julia
- **ビルドシステム**: Julia Package Manager (Pkg)
- **テストフレームワーク**: Julia Test.jl
- **ドキュメント**: Documenter.jl
- **主要ファイル**: `src/`

## 開発環境セットアップ

```julia
# Julia REPLで実行
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# テスト実行
using Pkg
Pkg.test()

# ドキュメントビルド
using Documenter
include("docs/make.jl")
```

## コード構造

```
src/
├── SparseIR.jl          # メインファイル
├── abstract.jl          # 抽象型定義
├── basis.jl             # 基底クラス
├── basis_set.jl         # 基底セット
├── C_API.jl             # C APIバインディング
├── dlr.jl               # DLR実装
├── freq.jl              # 周波数関連
├── kernel.jl            # カーネル関数
├── poly.jl              # 多項式クラス
├── sampling.jl          # サンプリング
└── sve.jl               # SVE実装
```

## 重要な注意事項

### 1. Juliaの型システム
- 型の不変性を考慮した設計
- パフォーマンスを重視した型安定性
- ジェネリックプログラミングの活用

### 2. メモリ管理
- `finalizer`を使用した適切なリソース管理
- C APIとの連携でのメモリリーク防止
- ガベージコレクションを考慮した設計

### 3. パフォーマンス
- 型安定性の確保
- 不要なアロケーションの回避
- ベクトル化された操作の活用

### 4. テスト
- 各関数の単体テスト
- 統合テスト
- パフォーマンステスト

## よく使用するコマンド

```julia
# パッケージの読み込み
using SparseIR

# テスト実行
using Pkg
Pkg.test()

# 特定のテストファイル実行
include("test/runtests.jl")

# ドキュメントビルド
using Documenter
include("docs/make.jl")

# パフォーマンステスト
using BenchmarkTools
@benchmark some_function()

# メモリ使用量確認
using Profile
@profile some_function()
```

## デバッグ

```julia
# デバッグモードで実行
julia --project=. -e "
using SparseIR
# デバッグコード
"

# プロファイリング
using Profile
@profile some_function()
Profile.print()

# メモリリークチェック
using Profile
@profile some_function()
Profile.print()
```

## ローカル libsparseir をテストで使う

`SparseIR.jl/src/C_API.jl` は環境変数 `SPARSEIR_LIB_PATH` が設定されている場合、そのパスのライブラリを優先してロードします。JLL の古いビルドでは未定義のシンボルがあることがあるため、ローカルビルドを使ってテストしたい場合は以下を実行してください。

### 重要な注意事項

**Juliaのプリコンパイルキャッシュの問題**: `C_API.jl` の `const libsparseir = get_libsparseir()` はモジュールロード時に一度だけ評価され、その結果がプリコンパイルキャッシュに保存されます。環境変数を設定しても、既にプリコンパイルされたキャッシュが存在する場合、古いライブラリパスが使われてしまいます。

**解決方法**: 環境変数を設定する前に、プリコンパイルキャッシュをクリアする必要があります。

### 手順

1) ライブラリのビルド（C++ 側）

```bash
cd libsparseir/backend/cxx/build
make -j
```

2) プリコンパイルキャッシュをクリア（重要）

```bash
# Juliaのバージョンに応じたキャッシュディレクトリを削除
rm -rf ~/.julia/compiled/v1.11/SparseIR
# または Julia のバージョンが異なる場合
julia --version  # バージョンを確認
rm -rf ~/.julia/compiled/v1.X/SparseIR  # v1.X を実際のバージョンに置き換え
```

3) 環境変数を設定してテスト実行

- macOS 例（.dylib）:

```bash
export SPARSEIR_LIB_PATH="$(pwd)/libsparseir/backend/cxx/build/libsparseir.dylib"
cd SparseIR.jl
julia --project -e 'using Pkg; Pkg.test()'
```

- Linux 例（.so）:

```bash
export SPARSEIR_LIB_PATH="$(pwd)/libsparseir/backend/cxx/build/libsparseir.so"
cd SparseIR.jl
julia --project -e 'using Pkg; Pkg.test()'
```

4) 使用中のライブラリ確認（任意）

```julia
julia --project -e "using SparseIR; using SparseIR.C_API; println(C_API.libsparseir)"
```

### 追加の注意事項

- `SPARSEIR_LIB_PATH` はパッケージ読込前（`using SparseIR` 前）に設定すること。
- ビルド生成物のファイル名は環境により `libsparseir.dylib`/`libsparseir.0.dylib`/`libsparseir.0.X.Y.dylib`（macOS）や `libsparseir.so`（Linux）などに変わるため、存在するパスを指定してください。
- プリコンパイルキャッシュをクリアしないと、環境変数の変更が反映されません。

## C APIとの連携

- C APIの関数は`C_API.jl`で定義
- エラーハンドリングは適切に実装
- 型変換は明示的に行う
- メモリ管理は`finalizer`で自動化

## リリース手順

1. バージョン番号を更新（`Project.toml`）
2. CHANGELOGを更新
3. テストが全て通ることを確認
4. ドキュメントをビルド
5. タグを作成してプッシュ
