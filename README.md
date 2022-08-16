# Adoption-Retirement-Analyzer

リポジトリ概要
------------------------------------------
ロジスティック回帰を用いた退職,内定を判定するプログラム.<br>
cubicのcsvデータを解析し,判定結果を出します.<br>
このバージョンでは/data/cubic_sample.csv を解析し,<br>
内定予想,退職予想の行を追加したcsvファイルを出力,判定結果を見れるようにしています.

環境
------------------------------------------
python 3.10.0<br>
scikit-learn 1.1.2<br>
pandas 1.4.3<br>
imbalanced-learn 0.9.1<br>

使い方
------------------------------------------
1.`git clone https://github.com/NANOBANA/Adoption-Retirement-Analyzer.git`

2.`cd Adoption-Retirement-Analyzer`

3.`python3 main.py`

モデルの精度について
------------------------------------------
いずれもテストデータを用いて

#### 内定 <br>
accuracy:77%, precision:11%, recall:97%, f1 score:20%

#### 退職 <br>
accuracy:87%, precision:71%, recall:100%, f1score:83%

となっています.

課題
-----------------------------------------
内定のモデル精度が悪いので,より良いモデルにする必要がある.
