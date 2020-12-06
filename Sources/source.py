import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import segmentation_models as sm
import numpy as np
from BatchManager import *
SM_FRAMEWORK=tf.keras



backbone = 'resnet50'
preprocess_input = sm.get_preprocessing(backbone)

data_manager = BatchManager(32 * 4, './dataset_small/data/images/', './dataset_small/data/masks/')
x_train, y_train = data_manager.get_data()

valid_manager = BatchManager(32 * 4, './dataset_small/valid/images/', './dataset_small/valid/masks/')
x_val, y_val = valid_manager.get_data()

print('Reading complete')

x_train = preprocess_input(x_train)
y_train = preprocess_input(y_train)

model = sm.Unet(backbone, classes=2, activation='softmax', encoder_weights='imagenet')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=1,
    epochs=4,
    validation_data=(x_val, y_val),
)

SaveModel(model, './model', '5')

