import cv2
import glob
import os
import numpy as np
from skimage.util import random_noise


class Data_Augmentation:

    def __init__(self, TARGET_FOLDER):
        self.dataset = []
        self.augmented_dataset = []
        self.TARGET_FOLDER = TARGET_FOLDER

    def load_data(self, IMAGE_FOLDER, LABEL_FOLDER):
        IMAGE_DIRS = glob.glob(os.path.join(IMAGE_FOLDER, "*"))
        # LABEL_DIRS = glob.glob(os.path.join(IMAGE_FOLDER,"*."+label_extension))
        for IMAGE_DIR in IMAGE_DIRS:
            data = {}
            img_name = IMAGE_DIR.split("/")[-1].split(".")[0]
            if (os.path.exists(os.path.join(LABEL_FOLDER, img_name+".txt"))):
                data["image"] = cv2.imread(IMAGE_DIR)
                data["bounding_boxes"] = self.load_label(os.path.join(LABEL_FOLDER, img_name+".txt"))
            self.dataset.append(data)

    def load_label(self, DIR):
        labels=[]
        with open(DIR) as f:
            for line in f:
                data_inline = line.strip(" ")
                label={
                    "class": data_inline[0],
                    "x_center": data_inline[1],
                    "y_center": data_inline[2],
                    "width": data_inline[3],
                    "height": data_inline[4]
                }
                labels.append(label)
        return labels


    def save_data(self):
        IMAGE_FOLDER = os.path.join(self.TARGET_FOLDER,"images")
        LABELS_FOLDER = os.path.join(self.TARGET_FOLDER, "labels")
        print("Saving data to " + self.TARGET_FOLDER + "...")
        for i, data in enumerate(self.augmented_dataset):
            
            pass

    def save_bb(self, labels, name):
        with open(name, "w") as f:
            for bb in labels:
                line=[bb["class"], lab]



    # def load_images(self, IMAGE_FOLDER, extension):
    #     IMAGE_DIRS = glob.glob(os.path.join(IMAGE_FOLDER, "*."+extension))
    #     for IMAGE_DIR in IMAGE_DIRS:
    #         self.images.append(cv2.imread(IMAGE_DIR))

    def noise(self, image):
        gaussian_img = random_noise(image, mode="gaussian")
        # gaussian_img = cv2.addWeighted(image,0.75,0.25*random_noise,0.25,0)
        gaussian_img = np.array(255*gaussian_img, dtype="uint8")
        print(gaussian_img)
        return gaussian_img
