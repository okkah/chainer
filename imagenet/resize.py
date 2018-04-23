from PIL import Image
import os

images = os.listdir("../data/mm_total_bmp") #ディレクトリのパス

for i in images:
    if i.endswith('.bmp'): #拡張子
        filename = "../data/mm_total_bmp/" + i
        img = Image.open(filename)
        print("Load {}".format(filename))
        img = img.resize((256, 256))
        img.save(filename) #上書き保存
    else: continue
