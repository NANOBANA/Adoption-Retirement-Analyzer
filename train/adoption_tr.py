import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# データの読み込み
train_data = pd.read_csv('../data/cubic_naitei_copy.csv')
test_data = pd.read_csv('../data/cubic_naitei_copy.csv')
data = pd.concat([train_data, test_data])

# 性別を[0,1]変換
def gender_convert(x):
    if x == '男':
        return 0
    elif x == '女':
        return 1

#在籍状況(入社3年以内)を変換
def enrolled_convert(x):
    if x == '0':
        return 0
    elif x == '内定':
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
train_data["性別"] = train_data["性別"].apply(gender_convert)
test_data["性別"] = test_data["性別"].apply(gender_convert)

#在籍情報を数値変換
train_data["内定有無"] = train_data["内定有無"].apply(enrolled_convert)
test_data["内定有無"] = test_data["内定有無"].apply(enrolled_convert)

#判定を数値変換
train_data["判定結果"] = train_data["判定結果"].apply(result_convert)
test_data["判定結果"] = test_data["判定結果"].apply(result_convert)


data = pd.concat([train_data, test_data])
# 説明変数X = df[["信頼係数","積極性","協調性","責任感","自己信頼性","指導性","共感性","感情安定性","従順性","自主性","達成欲求","ﾓﾗﾄﾘｱﾑ傾向","親和欲求","求知欲求","顕示欲求","秩序欲求","物質的欲望","危機耐性","自律欲求","支配欲求","勤労意欲","一般的"]]
X = pd.get_dummies(data.drop(["No.","内定有無","世代","ﾊﾟｰｿﾅﾘﾃｨｽｹｯﾁ","適合度"], axis=1))
# 目的変数
Y = data["内定有無"]

# 学習用データと検証用データに分割
train_size = len(train_data)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]

print('positive ratio = {:.2f}%'.format((len(Y_train[Y_train==1])/len(Y_train))*100))
#出力=> positive ratio = 2.88%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score


# ライブラリ
from imblearn.under_sampling import RandomUnderSampler

# クラス1の数を保存
count_train_class_one = Y_train.sum()
print('クラス1のサンプル数:{}'.format(count_train_class_one)) #クラス1のサンプル数表示

# クラス1が全体の10％になるまでクラス0を減らす
under = RandomUnderSampler(sampling_strategy={0:count_train_class_one*4, 1:count_train_class_one}, random_state=42)

# 学習用データに反映
x_train_under, y_train_under = under.fit_resample(X_train,Y_train)
print(x_train_under.shape) #学習用データのサンプル数確認
# ロジスティック回帰のインスタンス
lr = LogisticRegression(penalty='l2',          # 正則化項(L1正則化 or L2正則化が選択可能)
                           dual=False,            # Dual or primal
                           C=1.0,                 # 正則化の強さ
                           fit_intercept=True,    # バイアス項の計算要否
                           intercept_scaling=1,   # solver=‘liblinear’の際に有効なスケーリング基準値
                           class_weight=None,     # クラスに付与された重み
                           random_state=None,     # 乱数シード
                           solver='lbfgs',        # ハイパーパラメータ探索アルゴリズム
                           max_iter=100000,          # 最大イテレーション数
                           multi_class='auto',    # クラスラベルの分類問題（2値問題の場合'auto'を指定）
                           verbose=0,             # liblinearおよびlbfgsがsolverに指定されている場合、冗長性のためにverboseを任意の正の数に設定
                           warm_start=True,      # Trueの場合、モデル学習の初期化に前の呼出情報を利用
                           n_jobs=None,           # 学習時に並列して動かすスレッドの数
                           l1_ratio=None       # L1/L2正則化比率(penaltyでElastic Netを指定した場合のみ)
                          )
#モデル構築
lr.fit(x_train_under, y_train_under)


# 分類精度を検証
prob = lr.predict_proba(X_test)[:, 1] # 目的変数が1である確率を予測
pred = lr.predict(X_test) # 1 or 0 に分類
auc = roc_auc_score(y_true=Y_test, y_score=prob)
print('AUC = {:.2f}'.format(auc))
recall = recall_score(y_true=Y_test, y_pred=pred)
print('recall = {:.2f}'.format(recall))

# 確率算出の際は、predict_proba()メソッドを利用
Y_pred_proba = lr.predict_proba(X_test)

# データ出力
df_proba = pd.DataFrame()
df_proba["非生存率(Surviced=0)"] = Y_pred_proba[:,0]
df_proba["生存率(Surviced=1)"] = Y_pred_proba[:,1]

print(df_proba)
# 推論(0-1出力)
print(pred)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 正解率
print('accuracy: ',  round(accuracy_score(y_true=Y_test, y_pred=pred),2))
# 適合率
print('precision: ', round(precision_score(y_true=Y_test, y_pred=pred),2))
# 再現率
print('recall: ',    round(recall_score(y_true=Y_test, y_pred=pred),2))
# f1スコア
print('f1 score: ',  round(f1_score(y_true=Y_test, y_pred=pred),2))

import pickle
# モデルを保存
filename = 'naitei.pkl'
pickle.dump(lr,open(filename, 'wb'))


# テスト出力
test_data.to_csv('naitei_test.csv', encoding='utf_8_sig')
