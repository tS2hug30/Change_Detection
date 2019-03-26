# Change Detection
Multi-tusk U-netを用いた変化点検出のソースコード

## Constitution
* Multi_task_u-net
  * GT (ground truth用フォルダ)
  * Image_all (入力画像，ラベル画像用フォルダ)
  * unet (プログラム，出力結果用フォルダ)
  
* Image_all
  * Annotation (学習時におけるセグメンテーションのラベル画像用フォルダ)
  * Gray_ano (gray.pyの出力結果保存先フォルダ)
  * Original (学習時における入力画像用フォルダ)
  * Test (テスト時における入力画像，ラベル画像用フォルダ)
  * gray.py (ラベル画像を学習用に変換)
  
* unet (以下のディレクトリを追加)
  * *.py (各プログラム)
  * feat_visual (変化点検出タスクにおける出力結果保存先フォルダ)
    * 各閾値用のフォルダ(例："0","50","100")と"threshold"フォルダを**あらかじめ**作成する必要あり
  * OUTPUT (セグメンテーションタスクにおける出力結果保存先フォルダ)
    * テスト時に使用した画像の保存先："image0","image1"フォルダを**あらかじめ**作成する必要あり
    * セグメンテーション結果の保存先："result0","result1"フォルダを**あらかじめ**作成する必要あり
  * TRAIN (モデル保存先フォルダ)

## Requirement
* python 2.x
* cuda 8.0
* chainer 5.x
* OpenCV 3.x
* numpy


## Usage


#### 1. ラベル画像を学習用に変換
"Annotation"フォルダ内の変化前と変化後に分けられたフォルダ("t0","t1")に画像を入れておきます．  
`gray.py`に引数`t0`もしくは`t1`を与えて実行します．出力結果は，"Gray_ano"フォルダに保存されます．
> python gray.py t0  
> python gray.py t1

#### 2. ディレクトリの追加 (feat_visual)
"feat_visual"フォルダ内に閾値処理した2値画像を保存するディレクトリを作成します．(例："0","100","120")  
現在，ディレクトリ名を数値にしてあると仮定しています．  
また，閾値処理した可視化画像を保存するディレクトリ"threshold"を作成します．

#### 3. ディレクトリの追加 (OUTPUT)
"OUTPUT"フォルダ内にテスト時に使用した画像を保存するディレクトリ"image0","image1"を作成します．  
また，セグメンテーション結果を保存するディレクトリ"result0","result1"を作成します．

#### 4. 学習
`train.py`を実行します．学習を行う前に，あらかじめ，BATCH_SIZE，入力画像サイズ，使用するGPU番号，  
Data augmentationの有無の確認をお願いします．
> python train.py

もし，学習が止まってしまった場合，以下のコメントアウトを外し，再実行してください．
> serializers.load_npz('./TRAIN/' + '<モデル名>', model)  
> serializers.load_npz('./TRAIN/' + '<state名>', optimizer)  
> iteration = iteration + <モデル名，state名の拡張子前の数値>

#### 5. テスト
テストを行う前に，あらかじめ，入力画像サイズ，使用するGPU番号の確認をお願いします．  
`test.py`に引数を与えて実行します．指定されたパスにある指定された拡張子の画像すべてを用いてテストを行います．
> python test.py <モデル名> <閾値処理の値(0~250)>  

引数例
* モデル名：75000.model，90000.model
* 閾値値：0,10,20,30,40,50,・・・,240,250




