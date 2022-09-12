# Data Augmentation on YOLO

## Data augmentation techniques:
- Translation *
- Cropping 
- Noise *
- Brightness *
- Contrast *
- Saturation *
- Gaussian blur *

## Build virtual environment:
```bat
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```
## Run 
```bat
python3 main.py --images <IMAGES_FOLDER> --labels <LABELS_FOLDER> 
--output <OUTPUT_FOLDER> --nprocess <NUMBER_OF_AUGMENTED_IMAGES>
```