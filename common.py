import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

def imgShow(image):
    plt.figure(figsize=(15,2))
    plt.imshow(image, cmap="gray")
    plt.show()

def grayImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def autoResize(image, size = (800, 100)):
    gray = grayImage(image)
    height, width = gray.shape
    gray = cv2.resize(gray,(int(size[1]/height*width), size[1]))

    gray = np.pad(gray, ((0,0),(0, size[0]-gray.shape[1])), 'maximum')

    return gray

def orcPreprocess(image, default_fixed_size = (800, 100), default_filter_size = (10, 15), debug = False):
    img = autoResize(image, default_fixed_size)

    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if debug:
        imgShow(gray)

    blur = cv2.GaussianBlur(gray,(5,5),0)
    th1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # threshold the image using Otsu's thresholding method
    th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # apply a distance transform which calculates the distance to the
    # closest zero pixel for each pixel in the input image
    dist = cv2.distanceTransform(th2, cv2.DIST_L2, 5)
    
    # normalize the distance transform such that the distances lie in
    # the range [0, 1] and then convert the distance transform back to
    # an unsigned 8-bit integer in the range [0, 255]
    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = (dist * 255).astype("uint8")

    # threshold the distance transform using Otsu's method
    dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if debug:
        imgShow(dist)

    # apply an "opening" morphological operation to disconnect components
    # in the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)

    if debug:
        imgShow(opening)

    # find contours in the opening image, then initialize the list of
    # contours which belong to actual characters that we will be OCR'ing
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    chars = []
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # check if contour is at least 35px wide and 100px tall, and if
        # so, consider the contour a digit
        if w >= default_filter_size[0] and h >= default_filter_size[1]:
            chars.append(c)
    
    if len(chars) == 0:
        return np.zeros((), dtype = "uint8")

    # compute the convex hull of the characters
    chars = np.vstack([chars[i] for i in range(0, len(chars))])
    hull = cv2.convexHull(chars)

    # allocate memory for the convex hull mask, draw the convex hull on
    # the image, and then enlarge it via a dilation
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.dilate(mask, None, iterations=2)
    # take the bitwise of the opening image and the mask to reveal *just*
    # the characters in the image
    final = cv2.bitwise_and(opening, opening, mask=mask)

    if debug:
        imgShow(final)

    return final

CHAR_LIST = sorted("-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ")

# convert the words to array of indexs based on the char_list
def encodeText(txt):
    # encoding each output word into digits of indexes
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(CHAR_LIST.index(char))
        except:
            print("No found in char_list :", char)
        
    return dig_lst

def decodeText(arr):
    pred = ""
    for p in arr:  
        if int(p) >= 0:
            pred += CHAR_LIST[int(p)]
    return pred

# import padding library
from tensorflow.keras.preprocessing.sequence import pad_sequences
def pad_listints(txt, max_label_len = 20):
    return pad_sequences(txt, maxlen=max_label_len, padding='post', value = -1)

def pad_listint(txt, max_label_len = 20):
    return pad_listints([txt], max_label_len)[0]

from Levenshtein import distance as lev
def calCER(inp1, inp2):
    return lev(inp1, inp2)
