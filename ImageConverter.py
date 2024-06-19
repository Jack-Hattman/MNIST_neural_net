import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

def convert_input_png(png):

    img = Image.open(png)
    img = img.resize((28, 28), resample=PIL.Image.NEAREST)

    # Convert the image from the form (28, 28, 4) to (28, 28, 1)
    img = np.compress([False, False, True], img, axis=2)

    # Convert the image to (784, )
    img = img.flatten()

    img = img.tolist()

    return img

def center_image(img):

    img = img[0].reshape((28, 28))

    start_row = -1
    stop_row = -1

    start_col = 27
    stop_col = 0

    for i in range(len(img)):
        for j in range(len(img)):
            if img[i][j] != 0:
                if start_row == -1:
                    start_row = (i % 28)
                elif i > stop_row:
                    stop_row = (i % 28)

                if j < start_col:
                    start_col = j
                elif j > stop_col:
                    stop_col = j

    top_margin = start_row
    bottom_margin = (len(img) - 1) - stop_row

    left_margin = start_col
    right_margin = (len(img) - 1) - stop_col

    arb_scale = min((min((top_margin + bottom_margin), (left_margin + right_margin)) / 75), 1)

    if abs(top_margin - bottom_margin) > 1:
        if top_margin > bottom_margin:
            for i in range((top_margin - bottom_margin) // 2):
                img = np.delete(img, 0, axis=0)
                img = np.append(img, [np.zeros(28)], axis=0)
                top_margin -= 1
                bottom_margin += 1
        else:
            for i in range((bottom_margin - top_margin) // 2):
                img = np.delete(img, (len(img) - 1), axis=0)
                img = np.insert(img, 0, [np.zeros(28)], axis=0)
                top_margin += 1
                bottom_margin -= 1

    if abs(left_margin - right_margin) > 1:
        if left_margin > right_margin:
            for i in range((left_margin - right_margin) // 2):
                img = np.delete(img.T, 0, axis=0)
                img = np.append(img, [np.zeros(28)], axis=0).T
                left_margin -= 1
                right_margin += 1
        else:
            for i in range((right_margin - left_margin) // 2):
                img = np.delete(img.T, (len(img) - 1), axis=0)
                img = np.insert(img, 0, [np.zeros(28)], axis=0).T
                left_margin += 1
                right_margin -= 1

    if (top_margin + bottom_margin) < (left_margin + right_margin):

        smaller_margin = np.minimum(top_margin, bottom_margin)

        img = img[smaller_margin:(28 - smaller_margin)]
        img = img.T[smaller_margin:(28 - smaller_margin)].T

    else:

        smaller_margin = np.minimum(left_margin, right_margin)

        img = img[smaller_margin:(28 - smaller_margin)]
        img = img.T[smaller_margin:(28 - smaller_margin)].T


    img = Image.fromarray(img)

    img = img.resize((22, 22), resample=PIL.Image.BILINEAR)

    img = np.array(img)

    bg_img = np.zeros((28, 28))

    for i in range(3, 25):
        bg_img[i][3:25] = img[i - 3]

    img = bg_img

    img = img / 255

    img = np.round(img - arb_scale)

    img = img.reshape(784)

    return np.array([img])
