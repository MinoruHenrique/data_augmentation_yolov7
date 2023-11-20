import cv2
import glob
import os
import random
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path

class Data_Augmentation:

    def __init__(self, TARGET_FOLDER):
        self.dataset = []
        self.operations = []
        self.augmented_dataset = []
        self.TARGET_FOLDER = TARGET_FOLDER

    def load_data(self, IMAGE_FOLDER, LABEL_FOLDER):
        IMAGE_DIRS = glob.glob(os.path.join(IMAGE_FOLDER, "*"))
        # LABEL_DIRS = glob.glob(os.path.join(IMAGE_FOLDER,"*."+label_extension))
        for IMAGE_DIR in IMAGE_DIRS:
            data = {}
            img_name = Path(IMAGE_DIR).stem
            if (os.path.exists(os.path.join(LABEL_FOLDER, img_name+".txt"))):
                data["image"] = cv2.cvtColor(cv2.imread(IMAGE_DIR),cv2.COLOR_BGR2RGB)
                data["bounding_boxes"] = self.load_label(
                    os.path.join(LABEL_FOLDER, img_name+".txt"))
            self.dataset.append(data)

    def load_label(self, DIR):
        labels = []
        with open(DIR) as f:
            for line in f:
                data_inline = line.split(" ")
                label = {
                    "class": int(data_inline[0]),
                    "x_center": float(data_inline[1]),
                    "y_center": float(data_inline[2]),
                    "width": float(data_inline[3]),
                    "height": float(data_inline[4][:-1])
                }
                labels.append(label)
        return labels

    def run(self, n_processing=2):
        operations = [
            self.noise,
            self.translation,
            self.illumination,
            self.contrast,
            self.saturation,
            self.gaussian_blur
        ]
        i = 0
        for data in self.dataset:
            print(i)
            i += 1
            self.augmented_dataset.append(data)
            for i in range(n_processing):
                operation = random.choice(operations)
                self.operations.append(operation)
                new_data = operation(data)
                self.augmented_dataset.append(new_data)

    def save_data(self):
        IMAGE_FOLDER = os.path.join(self.TARGET_FOLDER, "images")
        LABELS_FOLDER = os.path.join(self.TARGET_FOLDER, "labels")
        if (not os.path.exists(self.TARGET_FOLDER)):
            os.mkdir(self.TARGET_FOLDER)
            os.mkdir(IMAGE_FOLDER)
            os.mkdir(LABELS_FOLDER)
        elif (not os.path.exists(IMAGE_FOLDER) or not os.path.exists(LABELS_FOLDER) ):
            os.mkdir(IMAGE_FOLDER)
            os.mkdir(LABELS_FOLDER)

        print('Saving data to "./' + self.TARGET_FOLDER + '"...')

        for i, data in enumerate(self.augmented_dataset):
            cv2.imwrite(os.path.join(
                IMAGE_FOLDER, str(i)+".jpg"), cv2.cvtColor(data["image"],cv2.COLOR_RGB2BGR))
            self.save_bb(os.path.join(LABELS_FOLDER, str(
                i)+".txt"), data["bounding_boxes"])

    def save_bb(self, DIR_NAME, labels):
        with open(DIR_NAME, "w+") as f:
            for bb in labels:
                line = "{} {} {} {} {}\n".format(
                    bb["class"],
                    bb["x_center"],
                    bb["y_center"],
                    bb["width"],
                    bb["height"]
                )
                f.write(line)

    def noise(self, data, lims=[0.008, 0.1]):
        new_data = {}
        image = data["image"]
        random_var =  np.random.uniform(lims[0],lims[1])
        gaussian_img = random_noise(image, mode="gaussian", var=random_var)
        # gaussian_img = cv2.addWeighted(image,0.75,0.25*random_noise,0.25,0)
        gaussian_img = np.array(255*gaussian_img, dtype="uint8")
        new_data["image"] = gaussian_img
        new_data["bounding_boxes"] = data["bounding_boxes"]
        return new_data

    def translation(self, data, lims=[-0.2, 0.3]):
        x = np.random.uniform(lims[0], lims[1])
        y = np.random.uniform(lims[0], lims[1])
        image = data["image"]
        bbs = data["bounding_boxes"]
        new_data = {}
        # image generation
        y_abs = int(y*image.shape[0])
        x_abs = int(x*image.shape[1])
        M = np.float32([
            [1, 0, x_abs],
            [0, 1, y_abs]
        ])
        new_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        # labels transformations
        new_bbs = []
        for bb in bbs:
            new_x = bb["x_center"]+x
            new_y = bb["y_center"]+y
            if (new_x > 1 or new_x < 0):
                continue
            if (new_y > 1 or new_y < 0):
                continue

            new_bb = {}
            new_bb["class"] = bb["class"]
            new_bb["x_center"] = new_x
            new_bb["y_center"] = new_y
            new_bb["width"] = bb["width"]
            new_bb["height"] = bb["height"]

            new_bbs.append(new_bb)

        new_data["image"] = new_image
        new_data["bounding_boxes"] = new_bbs

        return new_data

    def illumination(self, data, lims=[0.15, 3]):
        new_data = {}
        src = data["image"]
        gamma = np.random.uniform(lims[0], lims[1])
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        new_data["image"] = cv2.LUT(src, table)
        new_data["bounding_boxes"] = data["bounding_boxes"]
        return new_data

    def contrast(self, data,  lims=[0.5, 2]):
        new_data = {}

        random_enhance = np.random.uniform(lims[0], lims[1])        
        img = data["image"]
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(img)
        new_image = enhancer.enhance(random_enhance)
        new_data["image"] = np.array(new_image)
        new_data["bounding_boxes"] = data["bounding_boxes"]
        return new_data

    def saturation(self, data, lims=[0.5, 3]):
        new_data = {}
        random_enhance = np.random.uniform(lims[0], lims[1])
        img = data["image"]
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Color(img)
        new_image = enhancer.enhance(random_enhance)
        new_data["image"] = np.array(new_image)
        new_data["bounding_boxes"] = data["bounding_boxes"]

        return new_data

    def gaussian_blur(self, data, lims=[1.5, 4]):
        new_data = {}
        random_blur = np.random.uniform(lims[0], lims[1])
        img = data["image"]
        img = Image.fromarray(img)
        new_image = img.filter(ImageFilter.GaussianBlur(random_blur))
        new_data["image"] = np.array(new_image)
        new_data["bounding_boxes"] = data["bounding_boxes"]
        return new_data
