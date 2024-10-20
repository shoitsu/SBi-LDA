# SBi-LDA 実装

## 概要

このプロジェクトは、田村一樹さんらによる「評点付きレビュー文書を対象としたトピックモデルの構築に関する検討」で提案されたSBi-LDA（Switched Bi-directional Latent Dirichlet Allocation）モデルの実装です。

元論文: [評点付きレビュー文書を対象としたトピックモデルの構築に関する検討](https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=141419&item_no=1&attribute_id=1&file_no=1)

## SBi-LDAとは

SBi-LDAは、トピックモデルをレビュー（口コミ）データ用に拡張したモデルです。このモデルでは、以下の特徴があります：

- 商品ごとの共通のアイテムトピック
- ユーザーごとに共通の数のユーザートピック

詳細については、上記の論文を参照してください。

## 準備するデータ

以下のカラムを含むDataFrameを準備する必要があります：

- `review_text`：レビュー本文（形態素解析前の生のテキスト）
- `user_id`：レビュアーID
- `item_id`：商品ID

## パラメータの説明

モデルには以下のパラメータがあります：

- `n_user_topics`：ユーザートピック数
- `n_item_topics`：アイテムトピック数
- `alpha`, `beta`, `gamma`：事前分布のパラメータ（ハイパーパラメータ）
- `max_iter`：ギブスサンプリングの最大反復回数

## 実行方法

実行方法を含むJupyter Notebookファイル（`.ipynb`）とPythonファイル（`.py`）を用意しています。これらのファイルには、モデルの初期化、学習、および結果の可視化方法が記載されています。

1. Jupyter Notebookを使用する場合：
   - `SBi_LDA実装コード.ipynb` を開き、各セルを順に実行してください。

2. Pythonスクリプトを使用する場合：
   - コマンドラインで `python sbilda.py` を実行してください。

## ファイル構成

- `README.md`：本ファイル（プロジェクトの説明）
- `SBi_LDA実装コード.ipynb`：Jupyter Notebookファイル（実行例と詳細な説明）
- `sbilda.py`：Pythonスクリプトファイル（SBi-LDAの実装）
