# Change Detection
Multi-tusk U-netを用いた変化点検出のソースコード

## Constitution
- Multi_task_u-net
  - GT (ground truth用フォルダ)
  - Image_all (入力画像，ラベル画像用フォルダ)
  - unet (プログラム，出力結果用フォルダ)
  
- Image_all
  - Annotation (学習時におけるセグメンテーションのラベル画像用フォルダ)
  - Gray_ano (gray.pyの出力結果保存先フォルダ)
  - Original (学習時における入力画像用フォルダ)
  - Test (テスト時における入力画像，ラベル画像用フォルダ)
  - gray.py (ラベル画像を学習用に変換)



## Usage
