import cv2
import sys
import numpy as np
import random

def main(flags):
    # Read images from a video file in the current folder
    video_capture = cv2.VideoCapture("hw4.avi")  # Open video capture object
    got_image, bgr_image = video_capture.read()  # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()
    img_height = bgr_image.shape[0]
    img_width = bgr_image.shape[1]

    # start video writer
    if(flags['writeVideo']):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videoWriter = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=30.0,
                                      frameSize=(img_width, img_height))
    f = 675.0
    cx = 320.0
    cy = 240.0
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # set frame number to zero
    frame = 0
    # Read and show images until end of video is reached.
    framesWithDetected = 0
    while True:
        thresh_img = convertToBinary(bgr_image, flags)
        if (flags['showBinary']):
            cv2.imshow("binary image", thresh_img)

        corners, ids = getArucoMarkers(thresh_img)
        if ids is not None:
            framesWithDetected += 1
            rvecs, tvecs = getArucoPositions(bgr_image, k, corners, ids)
            for rvec, tvec, id in zip(rvecs, tvecs, ids):
                cv2.aruco.drawAxis(image=bgr_image, cameraMatrix=k, distCoeffs=None, rvec=rvec, tvec=tvec, length=4)
                scale = 2;
                pHeight = 4;
                pWidth = 1
                basePoints = [0, 0, 0]
                axes = np.matrix([
                    [basePoints[0] * scale - pHeight, basePoints[1] * scale + pWidth, basePoints[2] * scale + pWidth],
                    [basePoints[0] * scale - pHeight, basePoints[1] * scale + pWidth, basePoints[2] * scale - pWidth],
                    [basePoints[0] * scale - pHeight, basePoints[1] * scale - pWidth, basePoints[2] * scale - pWidth],
                    [basePoints[0] * scale - pHeight, basePoints[1] * scale - pWidth, basePoints[2] * scale + pWidth],
                    [basePoints[0] * scale, basePoints[1] * scale, basePoints[2] * scale],
                ])
                imagePoints, jacobian = cv2.projectPoints(np.float32(axes), rvec, tvec, k, None)
                bgr_image = cv2.line(bgr_image,
                                     tuple(np.int32(imagePoints[0]).ravel()),
                                     tuple(np.int32(imagePoints[1]).ravel()), (0, 0, 255), 1)
                bgr_image = cv2.line(bgr_image,
                                     tuple(np.int32(imagePoints[1]).ravel()),
                                     tuple(np.int32(imagePoints[2]).ravel()), (0, 0, 255), 1)
                bgr_image = cv2.line(bgr_image,
                                     tuple(np.int32(imagePoints[2]).ravel()),
                                     tuple(np.int32(imagePoints[3]).ravel()), (0, 0, 255), 1)
                bgr_image = cv2.line(bgr_image,
                                     tuple(np.int32(imagePoints[3]).ravel()),
                                     tuple(np.int32(imagePoints[0]).ravel()), (0, 0, 255), 1)
                for i in range(4):
                    bgr_image = cv2.line(bgr_image,
                                         tuple(np.int32(imagePoints[i]).ravel()),
                                         tuple(np.int32(imagePoints[4]).ravel()), (0, 0, 255), 1)
        if (flags['writeVideo']):
            videoWriter.write(bgr_image)
        frame += 1

        cv2.imshow("boxes", bgr_image)
        cv2.waitKey(1)
        # read video frames
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop
    videoWriter.release()
    print(framesWithDetected)


def getArucoMarkers(thresh_img):
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    # Detect a marker.  Returns:
    #   corners:   list of detected marker corners; for each marker, corners are clockwise)
    #   ids:   vector of ids for the detected markers
    corners, ids, _ = cv2.aruco.detectMarkers(
        image=thresh_img,
        dictionary=arucoDict
    )
    return corners, ids


def getArucoPositions(bgr_image, cameraMatrix, corners, ids):
    cv2.aruco.drawDetectedMarkers(
        image=bgr_image, corners=corners, ids=ids, borderColor=(0, 0, 255)
    )
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners=corners, markerLength=4,
        cameraMatrix=cameraMatrix, distCoeffs=None)
    return rvecs, tvecs


def convertToBinary(bgr_image, flags):
    hsv_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    planes = cv2.split(hsv_img)
    thresh_img = cv2.adaptiveThreshold(
        planes[2],
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=69,
        C=-3
    )
    return thresh_img


if __name__ == "__main__":
    main({
        'writeVideo' : True,
        'showBinary' : True
    })





