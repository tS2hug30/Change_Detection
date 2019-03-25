#-*- coding: utf-8 -*

import os
import cv2
import numpy as np
import sys
#入力画像のパス
img_path = "./Annotation/"
#入力画像ファイル指定
args = sys.argv
if (len(args) != 2) or ((args[1] != 't0') and (args[1] != 't1')):
    if len(args) != 2:
        print("Argument："+str(len(args)-1)+" != 1")
    if (args[1] != 't0') and (args[1] != 't1'):
        print("Argument is wrong")
    sys.exit(1)

def main(img_path,Arg):
        #画像の読み込み
        img_file = os.listdir(img_path+Arg+"/")
        #画像のソート
        img_file.sort()

        for num in range(len(img_file)):
                #画像の読み込み
                print(img_path+Arg+"/"+img_file[num])
                img = cv2.imread(img_path+Arg+"/"+img_file[num])
                h, w, c = img.shape
                #画像上をラスタスキャンし，画素値の指定を行う
                for y in range(h):
                        for x in range(w):
                                #画像上のBGRを取得
                                blue = img.item(y,x,0)
                                green = img.item(y,x,1)
                                red = img.item(y,x,2)
                                #グレースケール変換式
                                #gr = 0.299*red + 0.587*green + 0.114*blue

                                #空クラスの時、画素値指定20
                                if (blue == 180) and (green == 130) and (red == 70):
                                        img.itemset((y,x,0),20)
                                        img.itemset((y,x,1),20)
                                        img.itemset((y,x,2),20)
                                #道路クラスの時、画素値指定40
                                elif (blue == 232) and (green == 35) and (red == 243):
                                        img.itemset((y,x,0),40)
                                        img.itemset((y,x,1),40)
                                        img.itemset((y,x,2),40)
                                #その他クラスの時、画素値指定
                                else:
                                        img.itemset((y,x,0),0)
                                        img.itemset((y,x,1),0)
                                        img.itemset((y,x,2),0)

                if num < 9:
                    cv2.imwrite("./Gray_ano/"+Arg+"/"+"00"+str(num+1)+".bmp", img)  #出力画像
                elif num < 99:
                    cv2.imwrite("./Gray_ano/"+Arg+"/"+"0"+str(num+1)+".bmp", img)  #出力画像
                else:
                    cv2.imwrite("./Gray_ano/"+Arg+"/"+str(num+1)+".bmp", img)  #出力画像
                print(str(num+1)+"image was saved.")

if __name__ == '__main__':
        main(img_path,args[1])
