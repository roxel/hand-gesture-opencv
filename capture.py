from datetime import datetime
import math
import numpy as np
import cv2


NONE = (0, "not found")
WAVE_LEFT = (1, "wave left")
WAVE_RIGHT = (2, "wave right")
DOUBLE_TAP = (3, "double tap")
FIST = (4, "fist")
FINGERS_SPREAD = (5, "fingers spread")


def calculateFingers(shape, original):
    hull = cv2.convexHull(shape, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(shape, hull)
        if type(defects) is not None:
            count = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(shape[s][0])
                end = tuple(shape[e][0])
                far = tuple(shape[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    count += 1
                    cv2.circle(original, far, 8, [211, 84, 0], -1)
            return True, count
    return False, 0


def getBackgroundModel(camera):
    bgHistory = 2000
    bgThreshold = 16
    bgDetectShadows = False
    if camera.isOpened():
        backgroundModel = cv2.createBackgroundSubtractorMOG2(bgHistory, bgThreshold, bgDetectShadows)
        return backgroundModel
    return None


def removeBackground(frame, backgroundModel, show=False):
    foregroundMask = backgroundModel.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    foregroundMask = cv2.erode(foregroundMask, kernel, iterations=1)
    foregroundImage = cv2.bitwise_and(frame, frame, mask=foregroundMask)
    if show:
        cv2.imshow('foregroundImage', foregroundImage)
    return foregroundImage


def extractShape(foregroundImage, blurValue=61, threshold=30, show=False):
    grayedImage = cv2.cvtColor(foregroundImage, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(grayedImage, (blurValue, blurValue), 10)
    ret, extractedMask = cv2.threshold(blurredImage, threshold, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow('blur', blurredImage)
        cv2.imshow('extractedMask', extractedMask)
    return extractedMask


def processShape(extractedMask, original):
    _, contours, hierarchy = cv2.findContours(extractedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea, contourIndex = -1, None
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                contourIndex = i
        result = contours[contourIndex]
        hull = cv2.convexHull(result)

        cv2.drawContours(original, [result], 0, (0, 255, 0), 2)
        cv2.drawContours(original, [hull], 0, (0, 0, 255), 3)

        return result
    return None


def detect_hand_pose(shape, original):
    if shape is None:
        return NONE
    calculated, count = calculateFingers(shape, original)
    if calculated:
        if count == 0:
            return FIST
        elif 4 <= count <= 5:
            return FINGERS_SPREAD
    return NONE


def captureImage(camera, backgroundModel):
    ret, original = camera.read()

    original = cv2.bilateralFilter(original, 5, 50, 100)  # smoothing filter
    original = cv2.flip(original, 1)  # flip the frame horizontally for mirror effect

    hand_pose = NONE
    if backgroundModel:
        foregroundImage = removeBackground(original, backgroundModel)
        extractedMask = extractShape(foregroundImage)
        data = processShape(extractedMask, original)
        hand_pose = detect_hand_pose(data, original)
        cv2.imshow('original', original)
    return hand_pose


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    backgroundModel = getBackgroundModel(camera)
    bgModelIsSet = backgroundModel is not None
    while camera.isOpened() and bgModelIsSet:
        hand_pose = captureImage(camera, backgroundModel)
        hand_pose_name = hand_pose[1] if hand_pose else "not found"
        print("hand pose at %s: %s" % (str(datetime.now()), hand_pose_name))
        key = cv2.waitKey(100)
        if key == 27 or key == ord('q'):  # press ESC or 'q' to exit
            break
