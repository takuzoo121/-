import os
import pandas as pd
from  PIL import Image
import numpy as np
import sklearn

df_train = pd.read_csv("train.csv")#全て(25357個のデータ)
df_train2 = df_train.sample(500, random_state=0)#500個バージョン
DIR_IMAGES = "images"
IMG_SIZE = 100

X_train = []#配列
for i, row in df_train.iterrows():#pandasでデータの値を順番に取り出す
    img = Image.open(os.path.join(DIR_IMAGES, row.filename))#画像の取得
    img = img.crop((row.left, row.top, row.right, row.bottom))#画像の切り取り
    img = img.convert('L')#グレースケール化
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)#画像の大きさ揃える

    x = np.asarray(img, dtype=np.float)#数字の行列
    x = x.flatten()#ベクトル化
    X_train.append(x)#配列に入れ込む

X_train = np.array(X_train)#np配列にする
#X_train.shape#500個の訓練データの10000次元のベクトル

columns = ['company_name', 'full_name', 'position_name',
           'address', 'phone_number', 'fax',
           'mobile', 'email', 'url']
Y_train = df_train[columns].values#Y_train.shape#500個の訓練データの9個の正解ラベル
            
                    
#前処理
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()#変換器の初期化
scaler.fit(X_train)#開発データに合わせる,ないとエラー
X_scaled = scaler.transform(X_train)#標準化されたデータが返される

decomposer = PCA(n_components=30,  random_state=0)#圧縮先の次元数を指定
decomposer.fit(X_scaled)#使うデータに合わせる
X_pca = decomposer.transform(X_scaled)#PCAの結果を格            

##データ分割

#from sklearn.model_selection import train_test_split
#X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, train_size=0.8, random_state=0)
#X_dev2, X_val2, Y_dev2, Y_val2 = train_test_split(X_pca, Y_train, train_size=0.8, random_state=0)

##SVC
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics

estimator = SVC(C=0.01, kernel="rbf", gamma=0.001, probability=True)
classifier = OneVsRestClassifier(estimator)
classifier.fit(X_pca, Y_train)
#Y_val_pred = classifier.predict_proba(X_val2)#予測
#print("roc=",metrics.roc_auc_score(Y_val2, Y_val_pred, average='macro') )#評価

'''
##パイプライン(エラー出るからやめた)
from sklearn.pipeline import Pipeline

#steps = [('scaler', StandardScaler()),#標準化
#         ('decomposer', PCA( random_state=0)),#PCA(次元圧縮)
#         ('classifier', OneVsRestClassifier(SVC(probability=True)))]

#pipeline = Pipeline(steps)#まとめた

from sklearn.model_selection import GridSearchCV#交差検証によりハイパーパラメータの選択
from sklearn.metrics import make_scorer

params = {
	'C'         : [0.1, 0.01, 0.001],
	'kernel'  : ["linear", "rbf"],
	'gamma': [0.01, 0.001, 0.0001],
	'probability':[True] 
}

params2 = {
    "decomposer__n_components" : [ 45],
    "classifier__C"                             : [0.001], 
    "classifier__kernel"                      : ["linear"],#rbfにすると精度かなり下がる
#    "classifier__gamma"                    : [0.001]
}

scorer = make_scorer(metrics.roc_auc_score, average='macro', needs_proba=True)
estimator = SVC()
classifier = OneVsRestClassifier(estimator)

# グリッドサーチを行う --- (※4)
clf = GridSearchCV( classifier, params, cv=5, scoring=scorer)
clf.fit(X_dev2, Y_dev2)
print("学習器=", clf.best_estimator_)

pre = clf.predict_proba(X_val2)
#ac_score = metrics.accuracy_score(pre, Y_val2)
#print("正解率=",ac_score)
print("roc=",roc_auc_score(pre, Y_val2, average='macro'))
'''

##テスト
df_test = pd.read_csv("test.csv")
#df_test2 = df_test.sample(100, random_state=0)

X_test = []
for i, row in df_test.iterrows():
    img = Image.open(os.path.join(DIR_IMAGES, row.filename))
    img = img.crop((row.left, row.top, row.right, row.bottom))
    img = img.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
    
    x = np.asarray(img, dtype=np.float)
    x = x.flatten()
    X_test.append(x)
    
X_test = np.array(X_test)    
#X_test.shape#100個のテストデータの10000次元ベクトル

#scaler2 = StandardScaler()#変換器の初期化
#scaler2.fit(X_test)#開発データに合わせる,ないとエラー
X_test_scaled = scaler.transform(X_test)#標準化されたデータが返される

#decomposer2 = PCA(n_components=30,  random_state=0)#圧縮先の次元数を指定
#decomposer2.fit(X_scaled2)#使うデータに合わせる
X_test_pca = decomposer.transform(X_test_scaled)#PCAの結果を格   

Y_test_pred = classifier.predict_proba(X_test_pca)
np.savetxt('submission6.dat', Y_test_pred, fmt='%.6f')
