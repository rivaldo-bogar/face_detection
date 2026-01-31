import os
import random
import shutil

IMG_SRC = "dataset/images_all"
LBL_SRC = "dataset/labels_all"

IMG_TRAIN = "dataset/images/train"
IMG_VAL = "dataset/images/val"
LBL_TRAIN = "dataset/labels/train"
LBL_VAL = "dataset/labels/val"

os.makedirs(IMG_TRAIN, exist_ok=True)
os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_TRAIN, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

images = os.listdir(IMG_SRC)
random.shuffle(images)

split = int(0.8 * len(images))

for i, img in enumerate(images):
    label = img.replace(".jpg", ".txt")

    if i < split:
        shutil.move(f"{IMG_SRC}/{img}", f"{IMG_TRAIN}/{img}")
        shutil.move(f"{LBL_SRC}/{label}", f"{LBL_TRAIN}/{label}")
    else:
        shutil.move(f"{IMG_SRC}/{img}", f"{IMG_VAL}/{img}")
        shutil.move(f"{LBL_SRC}/{label}", f"{LBL_VAL}/{label}")
