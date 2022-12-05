from keras.models import load_model
import cv2 as cv
import numpy as np
from keras.applications.inception_v3 import preprocess_input


def crop(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        mask = gray_img > tol
        shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_cropping(img):
    image = crop(img)
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    h, w, depth = img.shape
    x = int(w / 2)
    y = int(h / 2)
    r = np.amin((x, y))
    circle = np.zeros((h, w), np.uint8)
    cv.circle(circle, (x, y), int(r), 1, thickness=-1)
    img = cv.bitwise_and(img, img, mask=circle)
    img = crop(img)
    img = cv.addWeighted(img, 4, cv.GaussianBlur(img, (0, 0), 10), -4, 128)
    return img


def preprocess_image(image_path):
    img = cv.imread(image_path)
    img = circle_cropping(img)
    img = cv.resize(img, (512, 512))
    return img


class DetectionModel:
    def process_image(self, image_url):
        # load pre-trained model
        new_model = load_model(r'D:\Diabetic Retinopathy Detection System\Detection_System\Detection_System\balanced_model_fine_tuned.h5')
        image = preprocess_image(image_url[1:])
        image = cv.resize(image, (224, 224)).astype(np.float32)
        img = np.expand_dims(image, axis=0)
        img = preprocess_input(img)
        classify = new_model.predict(img)
        results = np.argmax(classify, axis=1)
        labels = {0: 'Mild', 1: 'Moderate', 2: 'Normal', 3: 'Proliferate', 4: 'Severe'}
        return labels[results[0]]

