from matplotlib import pyplot as plt
import cv2

def draw_image(data, show=False):
    img = data["image"]
    dh, dw, _ = img.shape
    boxes=data["bounding_boxes"]
    for bb in boxes:
        x=bb["x_center"]
        y=bb["y_center"]
        w=bb["width"]
        h=bb["height"]
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 4)
    #print("x,y,w,h:",x,y,w,h)
    if(show):
        plt.figure()
        plt.imshow(img)
        plt.show()
    else:
        return img