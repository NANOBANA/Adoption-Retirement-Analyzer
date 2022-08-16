import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sys
import os
import shutil

# デバッグ用
print(os.getcwd())

try:
    os.mkdir(os.getcwd() + "\output")
except:
    pass

fp = sys.argv[1]
print(fp)
fn = os.path.basename(fp)
# データセット
df = pd.read_csv(fp)

df['内定有無'] = ''
df['No.'] = ''

# 性別を[0,1]変換
def gender_convert(x):
    if x == '男':
        return 0
    elif x == '女':
        return 1

def result_convert(x):
    if x == 'Ａ':
        return 0
    elif x == 'Ｂ':
        return 1
    elif x == 'Ｃ':
        return 2
    elif x == 'Ｄ':
        return 3
    elif x == 'Ｅ':
        return 4

#性別を数値変換
df["性別"] = df["性別"].apply(gender_convert)

#判定を数値変換
df["判定結果"] = df["判定結果"].apply(result_convert)

# 説明変数
X = df.drop(["No.","内定有無","世代","ﾊﾟｰｿﾅﾘﾃｨｽｹｯﾁ","適合度","個人ｺｰﾄﾞ","分類","区分","氏名"], axis=1)
# 目的変数
Y = df["内定有無"]
print(X)

#モデルをロードする
filename1 = 'models/naitei.pkl'
filename2 = 'models/retire.pkl'

loaded_model1 = pickle.load(open(filename1, 'rb'))
loaded_model2 = pickle.load(open(filename2, 'rb'))

result1 = loaded_model1.predict(X[:len(X)])
result2 = loaded_model2.predict(X[:len(X)])

df.insert(0, '内定予想', result1)
df.insert(1, '退職予想', result2)

def re_enrolled_convert(x):
    if x == 0:
        return "非内定"
    elif x == 1:
        return "内定"
def re_reitre_convert(x):
    if x == 0:
        return "在職"
    elif x == 1:
        return  "退職"

df["内定予想"] = df["内定予想"].apply(re_enrolled_convert)
df["退職予想"] = df["退職予想"].apply(re_reitre_convert)

print(df)

fn += 'out_file.csv'
df.to_csv(fn, encoding='utf_8_sig')
shutil.move(fn, 'output')
