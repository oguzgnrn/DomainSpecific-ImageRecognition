import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000
txtList = []
imgList = []
path = []#folder path
imgs = [f for f in glob.glob(path + "**/*.jpeg", recursive=True)]

for img in imgs:
    imgList.append(img)
print(imgList)
from PIL import Image

for i in range(len(imgList)):
    with open(txtList[i]) as txt:
        content = txt.read()
        icerikler = content.split(" ")
        image = Image.open(imgList[i])
        xorta = (image.width) * float(icerikler[1])
        xuzunluk = (image.width) * float(icerikler[3])
        yorta = (image.height) * float(icerikler[2])
        yuzunluk = (image.height) * float(icerikler[4])
        xcenter = xorta
        ycenter = yorta
        xbas = xcenter - (xuzunluk / 2)
        ybas = ycenter - (yuzunluk / 2)
        xson = xcenter + (xuzunluk / 2)
        yson = ycenter + (yuzunluk / 2)
        cropped = image.crop((xbas, ybas, xson, yson))
        cropped.show()
        cropped.save("resim_{}.jpeg".format(i + 1))
        print("Resim kaydediliyor {}".format(i + 1))
        print(imgList[i])
        print(image.width, image.height, xorta, xuzunluk, yorta, yuzunluk)
