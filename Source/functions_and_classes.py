import numpy as np
import os
import re
import cv2
from tensorflow.keras.models import model_from_json
import tqdm

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

class BatchManager:

	def __init__(self, image_dir, mask_dir, img_size, border):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.img_size = img_size
		self.border = border


	def split_image(self, img, frame_size=32 * 8, border=32):
		core_size = frame_size - 2 * border
		
		new_shape = list(img.shape)
		new_shape[0] += frame_size
		new_shape[1] += frame_size
		extended_img = np.empty(new_shape, img.dtype)
		s = (img.shape[0], img.shape[1])
		extended_img[border:border + s[0], border:border + s[1]] = img
		extended_img[border:border + s[0], 0:border] = extended_img[border:border + s[0], 2 * border:border:-1]
		extended_img[border:border + s[0], s[1] + border:] = extended_img[border:border + s[0], s[1] + border - 2:s[1] - core_size - 2:-1]
		extended_img[0:border, :] = extended_img[2 * border:border:-1, :]
		extended_img[s[0] + border:, :] = extended_img[s[0] + border - 2:s[0] - core_size - 2:-1, :]

		result = []
		for y in range(0, img.shape[0], core_size):
			for x in range(0, img.shape[1], core_size):
				result.append(extended_img[y:y + frame_size, x:x + frame_size])
		return result


	def generate_file(self, path, is_mask=False):
		if not is_mask:
			img_arr = cv2.imread(path, cv2.IMREAD_COLOR)
		else:
			img_arr = cv2.imread(path, 0)
		img_arr = img_arr.astype(np.float32)
		img_arr = cv2.resize(img_arr, (self.img_size, self.img_size)) / 255
		data = self.split_image(img_arr)
		return data


	def get_data(self):
		images = []
		masks = []
		n = 0

		for root, dirs, files in os.walk(self.image_dir):
			for file in files:
				images += self.generate_file(root + '/' + file)
				n += 1
		
		for root, dirs, files in os.walk(self.mask_dir):
			for file in files:
				masks += self.generate_file(root + '/' + file)

		#for folder in os.listdir(self.image_dir):
		#	images += generate_file(folder / im)
		#	masks += generate_file(folder / im, is_mask=True)
		#	n += 1
		return np.stack(images), np.stack(masks), n




def unsplit_image(array, output_shape, frame_size, border):
    core_size = frame_size - 2 * border
    assert core_size > 0

    new_shape = list(output_shape)
    new_shape[0] += core_size
    new_shape[1] += core_size
    extended = np.empty(new_shape, array[0].dtype)
    i = 0
    for y in range(0, output_shape[0], core_size):
        for x in range(0, output_shape[1], core_size):
            extended[y:y + core_size, x:x + core_size] = array[i][border:border + core_size, border:border + core_size]
            i += 1
    return extended[:output_shape[0], :output_shape[1]]


def unsplit_image_average(array, output_shape, frame_size, border):
    core_size = frame_size - 2 * border
    assert core_size > 0

    new_shape = list(output_shape)
    new_shape[0] += frame_size
    new_shape[1] += frame_size
    extended = np.empty(new_shape, np.float64)
    count = np.empty(new_shape, np.float64)
    i = 0
    for y in range(0, output_shape[0], core_size):
        for x in range(0, output_shape[1], core_size):
            extended[y:y + frame_size, x:x + frame_size] += array[i]
            count[y:y + frame_size, x:x + frame_size] += 1
            i += 1
    extended[count>0] /= count[count>0]
    return np.asarray(extended[border:border + output_shape[0], border:border + output_shape[1]], dtype=array[0].dtype)


def natural_sorting(a):
    a.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'(-{0,1}[0-9]+)|[^0-9]|', var)])
    return a


def generate_x_file(path, file, IMG_SIZE, BORDER):
	img_array = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
	img_array = cv2.resize(img_array, (32 * 8, 32 * 8))

	assert img_array is not None
	# plt.imshow(img_array), plt.show()
	data = split_image(img_array, IMG_SIZE, BORDER)
	return data


def generate_x_folder(path, IMG_SIZE, BORDER, img_postfix=".png"):
    training_data = []
    n = 0
    for img in natural_sorting(os.listdir(path)):
        if img_postfix is not None and img.endswith(img_postfix):
            training_data += generate_x_file(path, img, IMG_SIZE, BORDER)
            n += 1
    return np.stack(training_data), n


def split_image(img, frame_size=32 * 8, border=32):
    core_size = frame_size - 2 * border
    assert core_size > 0

    new_shape = list(img.shape)
    new_shape[0] += frame_size
    new_shape[1] += frame_size
    extended_img = np.empty(new_shape, img.dtype)
    s = (img.shape[0], img.shape[1])
    extended_img[border:border + s[0], border:border + img.shape[1]] = img
    extended_img[border:border + s[0], 0:border] = extended_img[border:border + s[0], 2 * border:border:-1]
    extended_img[border:border + s[0], s[1] + border:] = extended_img[border:border + s[0], s[1] + border - 2:s[1] - core_size - 2:-1]
    extended_img[0:border, :] = extended_img[2 * border:border:-1, :]
    extended_img[s[0] + border:, :] = extended_img[s[0] + border - 2:s[0] - core_size - 2:-1, :]
    # plt.imshow(extended_img); plt.show()

    result = []
    for y in range(0, img.shape[0], core_size):
        for x in range(0, img.shape[1], core_size):
            result.append(extended_img[y:y + frame_size, x:x + frame_size])
    return result