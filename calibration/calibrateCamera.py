import os

import cv2
import sys
import numpy as np
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
# for i in range(5):
#     board = cv2.aruco.CharucoBoard_create(2+i, 3+i, 10, 8, aruco_dict)
#     boardImg = board.draw((2000,2000))
#     cv2.imwrite(str(i+2)+"_"+str(i+3)+"_10_8_4x4_250.png", boardImg)

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
#some code taken from OpenCV documentation
def calibrate_charuco(marker_length, square_length):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    i = 4;
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard_create(2 + i, 3 + i, square_length, marker_length, aruco_dict)
    arucoParams = cv2.aruco.DetectorParameters_create()
    video_capture = cv2.VideoCapture("calibration.mp4")  # Open video capture object
    got_image, bgr_image = video_capture.read()  # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()
    counter, corners_list, id_list = [], [], []
    # Find the ArUco markers inside each image
    frame = 0
    while(True):
        #print(f'using frame {frame}')
        image = bgr_image
        img_gray = convertToBinary(image)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img_gray,
            aruco_dict,
            parameters=arucoParams
        )

        resp, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board
        )
        if charuco_corners is not None:
            for corner in charuco_corners:
                img_gray = cv2.circle(img_gray, (corner[0][0], corner[0][1]), 10, (255,0,255), -1)

        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        #print(resp)

        cv2.imshow("wein", img_gray)
        cv2.waitKey(1)
        if resp >= 10 and frame % 10 == 0:

            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
        got_image, bgr_image = video_capture.read()  # Make sure we can read video
        frame += 1;
        if not got_image:
            break;
    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_list,
        charucoIds=id_list,
        board=board,
        imageSize=img_gray.shape,
        cameraMatrix=None,
        distCoeffs=None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

# Dimensions in cm

SQUARE_LENGTH = 2.778125
MARKER_LENGTH = SQUARE_LENGTH * (8.0/10.0)

# # Calibrate
# ret, mtx, dist, rvecs, tvecs = calibrate_charuco(
#     MARKER_LENGTH,
#     SQUARE_LENGTH
# )
# Save coefficients into a file
# os.remove("calibration_charuco.yml")
# save_coefficients(mtx, dist, "calibration_charuco.yml")

# Load coefficients
mtx, dist = load_coefficients('calibration_charuco.yml')
video_capture = cv2.VideoCapture("calibration.mp4");
while(True):
    got_image, img = video_capture.read()
    if(not got_image):
        break
    i = 4;
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard_create(2 + i, 3 + i, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    arucoParams = cv2.aruco.DetectorParameters_create()
    img_bin = convertToBinary(img)
    corners, ids, rejected = cv2.aruco.detectMarkers(
        img_bin,
        aruco_dict,
        parameters=arucoParams
    )
    #refine markers?
    if ids is not None:
        resp, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_bin,
            board=board
        )
        if charucoIds is not None:
            img = cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, (0,0,255))
            _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners, charucoIds, board,
                cameraMatrix=mtx, distCoeffs=None, rvec=np.array([[0.0],[0.0],[0.0]]),tvec=np.array([[0.0],[0.0],[0.0]]))
            cv2.aruco.drawAxis(image=img, cameraMatrix=mtx, distCoeffs=None, rvec=rvec, tvec=tvec, length=4)
            scale = 2;
            pHeight = 4;
            pWidth = 10
            basePoints = [5, 5, 0]
            axes = np.matrix([
                [basePoints[0] * scale + pWidth, basePoints[1] * scale + pWidth, basePoints[2] * scale + pHeight],
                [basePoints[0] * scale + pWidth, basePoints[1] * scale - pWidth, basePoints[2] * scale + pHeight],
                [basePoints[0] * scale - pWidth, basePoints[1] * scale - pWidth, basePoints[2] * scale + pHeight],
                [basePoints[0] * scale - pWidth, basePoints[1] * scale + pWidth, basePoints[2] * scale + pHeight],
                [basePoints[0] * scale         , basePoints[1] * scale         , basePoints[2] * scale]
            ])
            imagePoints, jacobian = cv2.projectPoints(np.float32(axes), rvec, tvec, mtx, None)
            img = cv2.line(img,
                                 tuple(np.int32(imagePoints[0]).ravel()),
                                 tuple(np.int32(imagePoints[1]).ravel()), (0, 0, 255), 2)
            img = cv2.line(img,
                                 tuple(np.int32(imagePoints[1]).ravel()),
                                 tuple(np.int32(imagePoints[2]).ravel()), (0, 0, 255), 2)
            img = cv2.line(img,
                                 tuple(np.int32(imagePoints[2]).ravel()),
                                 tuple(np.int32(imagePoints[3]).ravel()), (0, 0, 255), 2)
            img = cv2.line(img,
                                 tuple(np.int32(imagePoints[3]).ravel()),
                                 tuple(np.int32(imagePoints[0]).ravel()), (0, 0, 255), 1)
            for i in range(4):
                img = cv2.line(img,
                                     tuple(np.int32(imagePoints[i]).ravel()),
                                     tuple(np.int32(imagePoints[4]).ravel()), (0, 0, 255), 4)
            # img = cv2.im
            cv2.imshow("v", img)
            cv2.waitKey(1)