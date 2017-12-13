import os
import pandas as pd
from  PIL import Image
import numpy as np
import sklearn 
#import visualize_tree 

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
#print(Y_train)
#データ作成↑

##not交差確認

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()#変換器の初期化
scaler.fit(X_train)#開発データに合わせる,ないとエラー
X_scaled = scaler.transform(X_train)#標準化されたデータが返される

decomposer = PCA(n_components=30,  random_state=0)#圧縮先の次元数を指定
decomposer.fit(X_scaled)#使うデータに合わせる
X_pca = decomposer.transform(X_scaled)#PCAの結果を格

#from sklearn.model_selection import train_test_split
#X_dev, X_val, Y_dev, Y_val = train_test_split(X_pca, Y_train, train_size=0.8, random_state=0)
#X_dev2, X_val2, Y_dev2, Y_val2 = train_test_split(X_train, Y_train, train_size=0.8, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics

estimator = RandomForestClassifier(n_estimators=40, max_features=9,  max_depth=20)#, probability=True)
classifier = OneVsRestClassifier(estimator)
#classifier.fit(X_dev, Y_dev)
classifier.fit(X_pca, Y_train)
#Y_val_pred = classifier.predict_proba(X_val)#予測
#print("roc=",metrics.roc_auc_score(Y_val, Y_val_pred, average='macro') )#評価

##テスト用
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
#decomposer2.fit(X_test_scaled)#使うデータに合わせる
X_test_pca = decomposer.transform(X_test_scaled)#PCAの結果を格納

Y_test_pred = classifier.predict_proba(X_test_pca)
np.savetxt('submission5.dat', Y_test_pred, fmt='%.6f')


