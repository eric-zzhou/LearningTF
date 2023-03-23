"""
Eric Zhou
Ms. Lewellen
TICS:AIML
March 9th, 2022

This website was referenced and a tiny bit of the code was copied:
https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/

This file is the OCR engine that actually does the character recognition using the ML model that was trained earlier.
It preprocess an image with text in it, find all the characters in the image, resize and add buffer to
each character to fit the input size, feed each character into the ML model, and gets the output.
"""

# Imports
import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import normalize

label_key = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Loading image
file_path = r'Final\test2.jpg'
og = cv2.imread(file_path)
# print(og)

# IMAGE PREPROCESSING
# Brightness and Contrast
chk = cv2.convertScaleAbs(og, alpha=1.5, beta=3)  # 1.8, 3 for first image
# Grayscale
chk = cv2.cvtColor(chk, cv2.COLOR_BGR2GRAY)
cv2.imwrite("Final\\gray.jpg", chk)
# Blur
bina = cv2.GaussianBlur(chk, (5, 5), 0)
# Binarization (convert everything to black and white)
bina = cv2.threshold(bina, 180, 255, cv2.THRESH_BINARY)[1]
# bina = cv2.threshold(bina, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite("Final\\threshold.jpg", bina)
# Change to only bounds
edged = cv2.Canny(bina, 30, 150)
cv2.imwrite("Final\\bounds.jpg", edged)


# Find all contours for letters based on black and white image
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]  # sort from left to right


# Getting all the characters
chars = []
for i in range(len(cnts)):
    # Find bounding box for the contour
    (x, y, w, h) = cv2.boundingRect(cnts[i])
    cv2.rectangle(og, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Combines all rectangles that overlap (based on x)
    edge_buffer = 0
    if chars:
        curr = chars[len(chars) - 1]
        if curr[1][0] + curr[1][2] + edge_buffer > x:
            temp = chars.pop(len(chars) - 1)[1]
            x2 = max(x + w, temp[0] + temp[2])
            y2 = max(y + h, temp[1] + temp[3])
            x = min(x, temp[0])
            y = min(y, temp[1])
            w = x2 - x
            h = y2 - y
            i -= 1

    # Extract the character and grab the width and height of the threshold image
    roi = chk[y:y + h, x:x + w]
    thresh = cv2.threshold(roi, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Resize along the longer edge
    (tH, tW) = thresh.shape
    if tW > tH:
        thresh = imutils.resize(thresh, width=28)
    else:
        thresh = imutils.resize(thresh, height=28)

    # Pad the edges to make it 28x28
    (tH, tW) = thresh.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)
    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    padded = cv2.resize(padded, (28, 28))

    # Prepare the padded image for model
    padded = padded.astype("float32")
    padded = normalize(padded, axis=1)
    padded = np.expand_dims(padded, axis=-1)

    # Update list
    chars.append((padded, (x, y, w, h)))

# Splits list into 2
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

# OCR the characters using our handwriting recognition model
model = load_model('Final\\ocrcnn.h5')
preds = model.predict(chars)

# Actual Output
output = ""
for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = label_key[i]
    output += label
    # cv2.rectangle(og, (x, y), (x + w, y + h), (255, 0, 0), 4)
    # cv2.putText(og, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imwrite("Final\\output.jpg", og)
print("output", output)
