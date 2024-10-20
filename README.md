# SBi-LDA
田村一樹さんらによる「評点付きレビュー文書を対象としたトピックモデルの構築に関する検討」で提案されたSBi-LDAを実装してみました。
元論文(https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=141419&item_no=1&attribute_id=1&file_no=1)

SBi-LDAとは
トピックモデルをレビュー(口コミ)データ用に拡張されたモデル。
商品ごとの共通のアイテムトピックとユーザーごとに共通の数をユーザートピックを仮定したモデル。
詳細は上記論文を参照下さい。

準備するデータ
以下のカラムを含むDataFrame
review_text：レビュー本文 形態素解析前の生のテキスト
user_id：レビュアーID
item_id：商品ID

パラメータの説明
n_user_topics：ユーザートピック数
n_item_topics：アイテムトピック数
alpha,beta,gamma：事前分布のパラメータ(ハイパーパラメータ)
max_iter：ギブスサンプリングの最大反復回数

実行方法まで載せたjpynbファイルとpythonファイルを用意しました。
