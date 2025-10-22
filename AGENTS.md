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
