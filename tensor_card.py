import os
import pandas as pd
from  PIL import Image
import numpy as np
import sklearn

df_train = pd.read_csv("train.csv")#全て(25357個のデータ)
df_train2 = df_train.sample(25000, random_state=0)#500個バージョン
DIR_IMAGES = "images"
IMG_SIZE = 64

X_train = []#配列
for i, row in df_train2.iterrows():#pandasでデータの値を順番に取り出す
    img = Image.open(os.path.join(DIR_IMAGES, row.filename))#画像の取得
    img = img.crop((row.left, row.top, row.right, row.bottom))#画像の切り取り
    img = img.convert('L')#グレースケール化
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)#画像の大きさ揃える(まだ画像データ)

    x = np.asarray(img, dtype=np.float)#数字の行列
    x = x.flatten()#ベクトル化
    X_train.append(x)#配列に入れ込む

X_train = np.array(X_train)#np配列にする
#X_train.shape#500個の訓練データの2500次元のベクトル

#ラベルデータ(出力となるデータ)
columns = ['company_name', 'full_name', 'position_name',
           'address', 'phone_number', 'fax',
           'mobile', 'email', 'url']
Y_train = df_train2[columns].values

#解が複数あるもの除去
use_num = []
for i in range(25000):
    if (np.sum(Y_train[i, ]) == 1):
        use_num.append(i)
        
X_train_reject = X_train[use_num, :] 
Y_train_reject = Y_train[use_num, :]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()#変換器の初期化
scaler.fit(X_train_reject.T)#開発データに合わせる,ないとエラー
X_train_scaled = scaler.transform(X_train_reject.T)#標準化されたデータが返される

from sklearn.model_selection import train_test_split
X_dev, X_val, Y_dev, Y_val = train_test_split(X_train_scaled.T, Y_train_reject, train_size=0.8, random_state=0)

import tensorflow as tf

#ここ変える
pixels = 64 * 64
nums = 9 

# プレースホルダを定義 --- (※2)
x  = tf.placeholder(tf.float32, shape=(None, pixels), name="x") # 画像データ
y_ = tf.placeholder(tf.float32, shape=(None, nums), name="y_")  # 正解ラベル 

# 重みとバイアスを初期化する関数 --- (※3)
def weight_variable(name, shape):
    W_init = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(W_init, name="W_"+name)
    return W

def bias_variable(name, size):
    b_init = tf.constant(0.1, shape=[size])
    b = tf.Variable(b_init, name="b_"+name)
    return b

# 畳み込みを行う関数 --- (※4)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 最大プーリングを行う関数 --- (※5)
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
        strides=[1,2,2,1], padding='SAME')
        
# 畳み込み層1 --- (※6)
with tf.name_scope('conv1') as scope:
    W_conv1 = weight_variable('conv1', [5, 5, 1, 32])#初期化
    b_conv1 = bias_variable('conv1', 32)#初期化
    #ここ変える
    x_image = tf.reshape(x, [-1, 64, 64, 1])#データ
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#畳み込み(活性化関数としてReLU)

# プーリング層1 ---- (※7)
with tf.name_scope('pool1') as scope:
    h_pool1 = max_pool(h_conv1)#畳み込み層1の結果を使う

#畳み込み層2 --- (※8)
with tf.name_scope('conv2') as scope:
    W_conv2 = weight_variable('conv2', [5, 5, 32, 64])#初期化
    b_conv2 = bias_variable('conv2', 64)#初期化
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#プーリング層の結果を使う(活性化関数ReLU)

# プーリング層2 --- (※9)
with tf.name_scope('pool2') as scope:
    h_pool2 = max_pool(h_conv2)

# 全結合レイヤー --- (※10)
with tf.name_scope('fully_connected') as scope:
    n = 16 * 16 * 64
    W_fc = weight_variable('fc', [n, 1024])
    b_fc = bias_variable('fc', 1024)
    h_pool2_flat = tf.reshape(h_pool2, [-1, n])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)        

# ドロップアウト(過剰適合)を排除 --- (※11)   #全結合ではなく一部を無視(過学習対策)
with tf.name_scope('dropout') as scope:
    keep_prob = tf.placeholder(tf.float32)   #dropoutする確率を変数
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)   #全結合レイヤーを入力にして上で決めた確率でdropout
    
# 読み出し層 --- (※12)
with tf.name_scope('readout') as scope:
    W_fc2 = weight_variable('fc2', [1024, 9])#初期化
    b_fc2 = bias_variable('fc2', 9)#初期化
    y_conv = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)#softmax関数でどの数字かを0,1符号化で    
    
# モデルの学習 --- (※13)
with tf.name_scope('loss') as scope:
    cross_entoropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    
with tf.name_scope('training') as scope:
    optimizer = tf.train.AdamOptimizer(1e-6)
    #train_step
    train_step = optimizer.minimize(cross_entoropy)#損失を最小にするように最小化

# モデルの評価 --- (※14)
with tf.name_scope('predict') as scope:
    predict_step = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))#予想したものがあってるか
    accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))#正解率
    
# feed_dictの設定 --- (※15)
def set_feed(images, labels, prob):
    return {x: images, y_: labels, keep_prob: prob}
    
# セッションを開始 --- (※16)
with tf.Session() as sess:
    #変数の初期化(おまじない)
    sess.run(tf.global_variables_initializer())
    
    # TensorBoardの準備(おまじない)
    #このファイルのあるフォルダでtensorboard --logdir=log_dirをコマンドラインでうつと見れる
    tw = tf.summary.FileWriter("log_dir", graph=sess.graph)
    
    #ここ変える
    # テスト用のフィードを生成(フィード　 = 代入する値)
    test_fd = set_feed(X_val, Y_val, 1)
    #test_fd = set_feed(X_val, Y_val, 1)
    
    M = int(X_dev.shape[0])#データ数
    N = int(50)#一つのバッチ
    
    # 訓練を開始 ---- (※17)
    for step in range(50000):
        #batch1 : 50個の(28,28)の画像データ
        #batch2 :  50個の(10,1)のラベルデータ
        
        #ランダムに取り出す
        num_set = np.random.randint(0,M,N)
        batch1 = X_dev[num_set, :]
        batch2 = Y_dev[num_set, :]
        
        fd = set_feed(batch1, batch2 , 0.5)
        
        _, loss = sess.run([train_step, cross_entoropy], feed_dict=fd)
        
        #2ステップごとに損失と正解率を出力
        if step % 1000 == 0:
            acc = sess.run(accuracy_step, feed_dict=test_fd)
            print("step=", step, "loss=", loss, "acc=", acc)
            
    # 最終結果を表示
    acc = sess.run(accuracy_step, feed_dict=test_fd)
    print("正解率=", acc)
                    

