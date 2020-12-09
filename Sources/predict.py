import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from BatchManager import *
import cv2
import numpy as np
# import math
# import matplotlib.pyplot as plt

DATADIR = "./test"
OUTPUTDIR = "./output"
OUTPUT_SHAPE = [32 * 8, 32 * 8, 3]
IMG_SIZE = 32 * 8

manager = BatchManager(256, 256, DATADIR)
x_train = manager.get_data()

model = LoadModel("./model", "5")


predict = model.predict(x_train)

# cv2.imshow('image', x_train[0])
# cv2.imshow('mask', predict[0])
# cv2.waitKey(0)
predict = predict * 255
cv2.imwrite('./test.png', predict[0])

print('done')
