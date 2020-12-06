import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



import tensorflow as tf
import segmentation_models as sm
import numpy as np
from functions_and_classes import *
SM_FRAMEWORK=tf.keras

backbone = 'resnet50'
preprocess_input = sm.get_preprocessing(backbone)


data_manager = BatchManager('./dataset/data/images/', './dataset/data/masks/', 32 * 8, 32 * 1)
x_train, y_train, _ = data_manager.get_data()

valid_manager = BatchManager('./dataset/valid/images/', './dataset/valid/masks/', 32 * 8, 32 * 1)
x_val, y_val, _ = valid_manager.get_data()

print('Reading complete')

x_train = preprocess_input(x_train)
y_train = preprocess_input(y_train)

model = sm.Unet(backbone, classes=1, activation='sigmoid', encoder_weights='imagenet')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=20,
    validation_data=(x_val, y_val),
)

SaveModel(model, './model', '5')

