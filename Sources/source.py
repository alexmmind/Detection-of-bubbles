import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import segmentation_models as sm
import numpy as np
from BatchManager import *
SM_FRAMEWORK=tf.keras



backbone = 'resnet50'
preprocess_input = sm.get_preprocessing(backbone)

data_manager = BatchManager(256, 256, './dataset/data/images/', './dataset/data/masks/')
x_train, y_train = data_manager.get_data()

valid_manager = BatchManager(256, 256, './dataset/valid/images/', './dataset/valid/masks/')
x_val, y_val = valid_manager.get_data()

print('Reading complete')

x_train = preprocess_input(x_train)
y_train = preprocess_input(y_train)

model = sm.Unet(backbone, classes=1, activation='sigmoid', encoder_weights='imagenet')

# dice_loss = sm.losses.DiceLoss()
# focal_loss = sm.losses.BinaryFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)

# metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# model.compile('Adam', total_loss, metrics)
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=20,
    validation_data=(x_val, y_val),
)

SaveModel(model, './model', '5')

