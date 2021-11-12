import cv2
import sys
import numpy as np
import random
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
def getCharucoBoard(x_dim, y_dim, square_size, marker_size, aruco_dict, offset):
    board = cv2.aruco.CharucoBoard_create(x_dim,y_dim,square_size,marker_size,aruco_dict)
    board.ids += offset*int((x_dim*y_dim)/2.0);
    return board



def main(flags):
    # TEST CODE TO GENERATE 5 DIFFERENT CHARUCO BOARDS
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    for i in range(4):
        board = getCharucoBoard(8, 11, 10, 8, aruco_dict, i)
        boardImg = board.draw((2000,2000))
        cv2.imwrite("test"+str(i)+".png", boardImg)
    cv2.waitKey(0)

    # set frame number to zero
    frame = 0

    framesWithDetected = 0
    k, dist = load_coefficients('calibration_charuco.yml')
    video_capture = cv2.VideoCapture("twoboardstest.mp4");

    #get first frame
    got_image, img = video_capture.read()
    img_height, img_width = img.shape[0], img.shape[1]

    # start video writer
    if(flags['writeVideo']):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videoWriter = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=30.0,
                                      frameSize=(img_width, img_height))

    while got_image:
        img_height, img_width = img.shape[0], img.shape[1]
        thresh_img = convertToBinary(img)
        if (flags['showBinary']):
            cv2.imshow("binary image", thresh_img)
        # create aruco board & dict
        SQUARE_LENGTH = 1.9 #2.778125
        MARKER_LENGTH = SQUARE_LENGTH * (8.0 / 10.0)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        board = getCharucoBoard(6, 7, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict, 0)
        arucoParams = cv2.aruco.DetectorParameters_create()

        img_bin = convertToBinary(img)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img_bin,
            aruco_dict,
            parameters=arucoParams
        )
        # refine markers?
        if ids is not None:
            resp, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=img_bin,
                board=board
            )
            if charucoIds is not None:
                img = cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, (0, 0, 255))
                _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charucoCorners, charucoIds, board,
                    cameraMatrix=k, distCoeffs=None, rvec=np.array([[0.0], [0.0], [0.0]]),
                    tvec=np.array([[0.0], [0.0], [0.0]]))
                cv2.aruco.drawAxis(image=img, cameraMatrix=k, distCoeffs=None, rvec=rvec, tvec=tvec, length=4)
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
        SQUARE_LENGTH = 1.9  # 2.778125
        MARKER_LENGTH = SQUARE_LENGTH * (8.0 / 10.0)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        board = getCharucoBoard(6, 7, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict, 1)
        arucoParams = cv2.aruco.DetectorParameters_create()

        img_bin = convertToBinary(img)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img_bin,
            aruco_dict,
            parameters=arucoParams
        )
        # refine markers?
        if ids is not None:
            resp, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=img_bin,
                board=board
            )
            if charucoIds is not None:
                img = cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, (0, 0, 255))
                _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charucoCorners, charucoIds, board,
                    cameraMatrix=k, distCoeffs=None, rvec=np.array([[0.0], [0.0], [0.0]]),
                    tvec=np.array([[0.0], [0.0], [0.0]]))
                cv2.aruco.drawAxis(image=img, cameraMatrix=k, distCoeffs=None, rvec=rvec, tvec=tvec, length=4)
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
        if (flags['writeVideo']):
            videoWriter.write(img)
        cv2.imshow("eee", img)
        cv2.waitKey(30)
        frame += 1
        got_image, img = video_capture.read()
    videoWriter.release()
    print(framesWithDetected)



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
        'showBinary' : True
    })





