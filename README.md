# Adoption-Retirement-Analyzer

リポジトリ概要
------------------------------------------
ロジスティック回帰を用いた退職,内定を判定するプログラム.<br>
cubicのcsvデータを解析し,内定予想,退職予想の行を追加したcsvファイルを出力し<br>
判定結果が簡単に見れるようになっています.

環境
------------------------------------------
python 3.10.0<br>
scikit-learn 1.1.2<br>
pandas 1.4.3<br>
imbalanced-learn 0.9.1<br>

ディレクトリ構造
------------------------------------------
├── README.md
├── data
│   ├── cubic_naitei_copy.csv
│   ├── cubic_retire.csv
│   └── cubic_sample.csv
├── lib.bat
├── main.bat
├── main.py
├── models
│   ├── naitei.pkl
│   └── retire.pkl
└── train
    ├── adoption_tr.py
    └── retire_tr.py
    
各ディレクトリについて
------------------------------------------
#### data/ <br>
実験用csvが格納されているディレクトリ.

#### models/ <br>
機械学習のモデルデータが入っているディレクトリ.

#### tain/ <br>
機械学習のモデルデータ作成プログラムに関するディレクトリ.

使い方
------------------------------------------
## Mac OS (Linux) <br>
1.`git clone https://github.com/NANOBANA/Adoption-Retirement-Analyzer.git`

2.`cd Adoption-Retirement-Analyzer`

3.`python3 main.py cubic_file.csv`

## Windows <br>
1.`git clone https://github.com/NANOBANA/Adoption-Retirement-Analyzer.git`

2.`cd Adoption-Retirement-Analyzer`

3.drag and drop to main.bat


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
