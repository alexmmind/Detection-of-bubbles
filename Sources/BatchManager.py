import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import tqdm

class BatchManager:
    def __init__(self, image_size1, image_size2, path_image, path_mask=''):
        self.path_image = path_image
        self.path_mask = path_mask
        self.image_size = (image_size1, image_size2)

    def read_file(self, path, is_mask=False):
        if is_mask:
            mask = cv2.imread(path, 0)
            mask = cv2.resize(mask, self.image_size)
            mask = mask > 100
            # mask = [mask.astype(np.float32)] # может быть, нужно сделать список из одного элемента
            mask = np.reshape(mask, self.image_size)
            # cv2.imshow('mask', mask[0])
            return mask

        else:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = image.astype(np.float32)
            image = cv2.resize(image, self.image_size)
            # cv2.imshow('image', image)
            
            return image

    def get_data(self):
        images = []
        for root, dirs, files in os.walk(self.path_image):
            for file in files:
                images.append(self.read_file(root + '/' + file))

        if self.path_mask != '':
            masks = []
            for root, dirs, files in os.walk(self.path_mask):
                for file in files:
                    masks.append(self.read_file(root + '/' + file, is_mask=True))
            return np.stack(images), np.stack(masks)
        
        return np.stack(images)



def SaveModel(model, path, name):
    with open(path + '/' + name + '.json', 'w') as json_file:
        model_json = model.to_json()
        json_file.write(model_json)
    model.save_weights(path + '/' + name + '.h5')

def LoadModel(path, name):
    with open(path + "/" + name + ".json", "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(path + "/" + name + ".h5")
    return model

