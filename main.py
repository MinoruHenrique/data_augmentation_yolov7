import argparse
from utils.dataaugmentation import Data_Augmentation
from utils import datavisualization

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--images", help="Folder that contains dataset images")
    parser.add_argument(
        "-l", "--labels", help="Folder that contains dataset labels")
    parser.add_argument(
        "-o", "--output", help="Folder where augmented data files are saved")
    parser.add_argument(
        "-n", "--nprocess", help="Number of new images that will be generated")
    args=parser.parse_args()
    data_augmentation = Data_Augmentation(args.output)
    data_augmentation.load_data(
        args.images, args.labels)
    data_augmentation.run(n_processing=int(args.nprocess))
    data_augmentation.save_data()
