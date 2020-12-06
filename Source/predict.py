import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from functions_and_classes import *
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

DATADIR = "./test"
OUTPUTDIR = "./output"
OUTPUT_SHAPE = [32 * 8, 32 * 8, 3]
IMG_SIZE = 32 * 8
BORDER = 32 * 1
x_train, img_n = generate_x_folder(DATADIR, IMG_SIZE=IMG_SIZE, BORDER=BORDER, img_postfix=".bmp")
# x_train, y_train, img_n = BatchManager.generate_xy_folder(DATADIR, IMG_SIZE=IMG_SIZE, BORDER=BORDER, img_postfix="original.png", mask_postfix="fimal_mask.png")

model = LoadModel("./model", "5")

# x_train = np.float32(x_train)
predict = model.predict(x_train)


active_bubbles = []
history = []
last_id = 0
N = x_train.shape[0] // img_n
for i in range(x_train.shape[0] // N):
    orig = unsplit_image(x_train[i*N:(i+1)*N], output_shape=OUTPUT_SHAPE, frame_size=IMG_SIZE, border=BORDER)
    mask = np.argmax(unsplit_image(predict[i*N:(i+1)*N], output_shape=OUTPUT_SHAPE, frame_size=IMG_SIZE, border=BORDER), 2)
    cv2.imwrite('fsa.bmp', mask)
    cv2.imwrite('fsa1.bmp', orig)


print('done')