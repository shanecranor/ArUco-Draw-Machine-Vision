import time

import cv2
import sys
import numpy as np
import random
import calibration
def getCharucoBoard(x_dim, y_dim, square_size, marker_size, aruco_dict, offset):
    board = cv2.aruco.CharucoBoard_create(x_dim,y_dim,square_size,marker_size,aruco_dict)
    board.ids += offset*int((x_dim*y_dim)/2.0);
    return board




def main(flags):
    # TEST CODE TO GENERATE 5 DIFFERENT CHARUCO BOARDS
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    # for i in range(4):
    #     board = getCharucoBoard(8, 11, 10, 8, aruco_dict, i)
    #     boardImg = board.draw((2000,2000))
    #     cv2.imwrite("test"+str(i)+".png", boardImg)
    # cv2.waitKey(0)

    # set frame number to zero
    frame = 0

    framesWithDetected = 0
    k, dist = calibration.load_coefficients('calibration_charuco.yml')
    video_capture = cv2.VideoCapture("4boardtest.mp4");

    #get first frame
    got_image, img = video_capture.read()
    img_height, img_width = img.shape[0], img.shape[1]

    # start video writer
    if(flags['writeVideo']):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videoWriter = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=30.0,
                                      frameSize=(img_width, img_height))
    linePoints = [];
    oldCanvasVecs = [None, None, None, None]
    oldFt = 0
    while got_image:
        ft = time.time()
        thresh_img = convertToBinary(img)
        # create aruco board & dict
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        arucoParams = cv2.aruco.DetectorParameters_create()
        #detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(thresh_img, aruco_dict, parameters=arucoParams, cameraMatrix=k)
        canvasVecs = [None, None, None, None]
        drawing = False
        for i in range(4):
            canvasVecs[i] = None
            SQUARE_LENGTH = 1.9 #2.778125
            MARKER_LENGTH = SQUARE_LENGTH * (8.0 / 10.0)
            board = getCharucoBoard(8, 11, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict, i)
            newCorners, newIds, newRejected, recovered = cv2.aruco.refineDetectedMarkers(thresh_img, board, corners, ids,
                                                                                rejected, cameraMatrix=k)
            if ids is not None:
                resp, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                    newCorners, newIds, thresh_img, board, cameraMatrix=k
                )
                if charucoIds is not None:
                    #img = cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, (0, 0, 255))
                    _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charucoCorners, charucoIds, board,
                        cameraMatrix=k, distCoeffs=None, rvec=np.array([[0.0], [0.0], [0.0]]),
                        tvec=np.array([[0.0], [0.0], [0.0]]))
                    if i == 2:
                        #img = drawPyramid(img, k, rvec, tvec, (255,0,0))
                        linePoints.append([tvec, (0,255,0)])
                        drawing = True
                    else:
                        #img = drawPyramid(img, k, rvec, tvec)
                        canvasVecs[i] = tvec
        if not drawing:
            linePoints.append(None)
        #calculate movement in all the canvas markers
        totalDelta, deltaCount = calculateCameraMovement(canvasVecs, oldCanvasVecs)
        pts = len(linePoints)

        if pts > 1:
            if not pointIsValid(img, k, linePoints, pts):
                linePoints = linePoints[:-1]
            updateLinePosition(linePoints, totalDelta)
            img = drawLines(img, linePoints, k, rvec)
        #only update the old canvas data if the new canvas data isn't empty!
        if not all(x is None for x in canvasVecs):
            oldCanvasVecs = canvasVecs

        if flags['showFPS']:
            fps = 1 / (ft - oldFt)
            cv2.putText(img, str(round(fps,2)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        oldFt = ft
        if flags['writeVideo']:
            videoWriter.write(img)
        if flags['showBinary']:
            cv2.imshow("binary image", thresh_img)
        cv2.imshow("live output", img)
        cv2.waitKey(1)
        frame += 1
        got_image, img = video_capture.read()
    if flags['writeVideo']:
        videoWriter.release()
    print(framesWithDetected)


def pointIsValid(img, k, linePoints, pts):
    if (pts > 1 and linePoints[pts - 1] is not None):
        if isOffScreen(img, linePoints[pts - 1][0], k) or np.linalg.norm(linePoints[pts - 1][0]) == 0.0:
            return False
        # if linePoints[pts - 2] is not None:
        #     print(np.linalg.norm(linePoints[pts - 1][0] - linePoints[pts - 2][0]))
        #     dis = np.linalg.norm(linePoints[pts - 1][0] - linePoints[pts - 2][0])
        #     maxDist = 100
        #     if dis > maxDist or dis == 0.0:
        #         return False
    return True


def isOffScreen(img, point, k):
    axes = np.matrix([[0, 0, 0]])
    pts, jacobian = cv2.projectPoints(np.float32(axes), np.array([[0.0], [0.0], [0.0]]), point, k, None)
    s = img.shape
    pts = pts[0][0]
    return pts[0] > s[0] or pts[1] > s[1] or pts[0] < 0 or pts[1] < 0
def updateLinePosition(pointsArr, totalDelta):
    for p in range(len(pointsArr) - 1):
        if pointsArr[p] is not None:
            pointsArr[p][0] -= totalDelta


def calculateCameraMovement(canvasVecs, oldCanvasVecs):
    totalDelta = 0
    deltaCount = 0
    for i in range(4):
        if oldCanvasVecs[i] is not None and canvasVecs[i] is not None:
            currentDelta = oldCanvasVecs[i] - canvasVecs[i]
            totalDelta += currentDelta
            deltaCount += 1
    if deltaCount != 0:
        totalDelta = totalDelta / deltaCount
    else:
        totalDelta = 0
    return totalDelta, deltaCount


def drawLines(img, pointsArr, k, rvec):
    axes = np.matrix([[0,0,0]])
    for i in range(len(pointsArr) - 1):
        #check if there is a gap, if there is, skip to end of loop
        if pointsArr[i] is not None and pointsArr[i + 1] is not None:
            imagePoints0, jacobian = cv2.projectPoints(np.float32(axes), rvec, pointsArr[i][0], k, None)
            imagePoints1, jacobian = cv2.projectPoints(np.float32(axes), rvec, pointsArr[i + 1][0], k, None)
            img = cv2.line(img,
                           tuple(np.int32(imagePoints0).ravel()),
                           tuple(np.int32(imagePoints1).ravel()), pointsArr[i][1], 2)
    return img

def drawPyramid(img, k, rvec, tvec, color=(0,0,255)):
    scale = 2;
    pHeight = 4;
    pWidth = 10
    basePoints = [5, 5, 0]
    axes = np.matrix([
        [basePoints[0] * scale + pWidth, basePoints[1] * scale + pWidth, basePoints[2] * scale + pHeight],
        [basePoints[0] * scale + pWidth, basePoints[1] * scale - pWidth, basePoints[2] * scale + pHeight],
        [basePoints[0] * scale - pWidth, basePoints[1] * scale - pWidth, basePoints[2] * scale + pHeight],
        [basePoints[0] * scale - pWidth, basePoints[1] * scale + pWidth, basePoints[2] * scale + pHeight],
        [basePoints[0] * scale, basePoints[1] * scale, basePoints[2] * scale]
    ])
    imagePoints, jacobian = cv2.projectPoints(np.float32(axes), rvec, tvec, k, None)
    img = cv2.line(img,
                   tuple(np.int32(imagePoints[0]).ravel()),
                   tuple(np.int32(imagePoints[1]).ravel()), color, 2)
    img = cv2.line(img,
                   tuple(np.int32(imagePoints[1]).ravel()),
                   tuple(np.int32(imagePoints[2]).ravel()), color, 2)
    img = cv2.line(img,
                   tuple(np.int32(imagePoints[2]).ravel()),
                   tuple(np.int32(imagePoints[3]).ravel()), color, 2)
    img = cv2.line(img,
                   tuple(np.int32(imagePoints[3]).ravel()),
                   tuple(np.int32(imagePoints[0]).ravel()), color, 1)
    for i in range(4):
        img = cv2.line(img,
                       tuple(np.int32(imagePoints[i]).ravel()),
                       tuple(np.int32(imagePoints[4]).ravel()), color, 4)
    return img


def convertToBinary(bgr_image):
    hsv_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    planes = cv2.split(hsv_img)
    thresh_img = cv2.adaptiveThreshold(
        planes[2],
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=699,
        C=-15
    )
    return thresh_img


if __name__ == "__main__":
    main({
        'writeVideo' : False,
        'showBinary' : False,
        'showFPS' : True
    })





