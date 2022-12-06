import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
import tensorflow
import camera_matrix_calculation


from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator()

gen = idg.flow_from_directory("./imgs",target_size=(200,200))

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout

inlayer = Input(shape=(200,200,3))
f1 = Conv2D(8, 3, activation="relu")(inlayer)
t1 = MaxPooling2D(2)(f1)
flat = Flatten()(inlayer)
a2 = Dense(500, activation="relu")(flat)
drop1 = Dropout(.2)(a2)
a5 = Dense(100, activation="relu")(a2)
out_layer = Dense(2, activation="softmax")(a5)

model = Model(inlayer, out_layer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(gen, epochs=10)

filename ="Derrick.mov"
capture = cv2.VideoCapture(filename)


def getClassName(index):
    if index == 0:
        return "object found"
    else:
        return "object not found"

font = cv2.FONT_HERSHEY_COMPLEX

org = (50, 50)

cam_matx = camera_matrix_calculation.camera_matx

cam_matx_inv = np.linalg.inv(cam_matx)


foundImg = cv2.imread("./thepictures/found/capture_isp_1.png")
res,coords = model.predict(np.array([foundImg]))

(fX, fY, fW, fH) = coords

size = "(" + str(fW) + "," + str(fH) + ")"


dimension_matrix = cam_matx_inv.dot(project_points)
size = "(" + str(-1*dimension_matrix[0][0]) + "," + str(-1*dimension_matrix[1][0]) + ")"


fontScale = 1

color = (255, 0, 0)  

thickness = 2

def img_alignment(img1, img2):
    img1, img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
    img_size = img1.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3,dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    
    n_iterations = 5000
    termination_eps = 1e-10

    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, n_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img2_aligned = cv2.warpAffine(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return img2_aligned

while True:
    _, img1 = capture.read()
    _, img2 = capture.read()

    res = model.predict(np.array([cv2.resize(img1, (200,200))]))

    img1 = cv2.putText(img1, getClassName(res.argmax(axis=1)) , org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    if getClassName(res.argmax(axis=1))=="object found":
        img1 = cv2.putText(img1,size,(100,100), font,fontScale, color, thickness, cv2.LINE_AA)

    diff = cv2.absdiff(img1, img2)
    
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff_blur = cv2.GaussianBlur(diff_gray, (5,5,), 0)

    _, binary_img = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, b, l = cv2. boundingRect(contour)
        if cv2.contourArea(contour) > 300:
            cv2.rectangle(img1, (x, y), (x+b, y+l), (0,255,0), 2)
    
    cv2.imshow("Motion", img1)
    key = cv2.waitKey(1)
    if key%256 == 27:
        print("Closing program")
        exit()


