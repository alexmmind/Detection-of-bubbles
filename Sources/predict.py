import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from BatchManager import *
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

DATADIR = "./test"
OUTPUTDIR = "./output"
OUTPUT_SHAPE = [32 * 8, 32 * 8, 3]
IMG_SIZE = 32 * 8

manager = BatchManager(IMG_SIZE, DATADIR, './mask')
x_train, _ = manager.get_data()

model = LoadModel("./model", "5")

x_train = np.float32(x_train)
predict = model.predict(x_train)


cv2.imshow('image', predict[0])
cv2.waitKey(0)
predict = predict * 255
cv2.imwrite(DATADIR + '/test.png', predict[0])
print(predict.shape)

print('done')