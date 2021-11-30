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
	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
	# for i in range(4):
	# 	board = getCharucoBoard(5, 7, 10, 8, aruco_dict, i)
	# 	boardImg = board.draw((2000,2000))
	# 	cv2.imwrite("testLarge"+str(i)+".png", boardImg)
	# cv2.waitKey(0)

	# set frame number to zero
	frame = 0
	#get first frame
	video_capture = cv2.VideoCapture("cube.mp4");
	got_image, img = video_capture.read()
	img_height, img_width = img.shape[0], img.shape[1]

	# start video writer
	if(flags['writeVideo']):
		fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
		videoWriter = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=30.0,
									  frameSize=(img_width, img_height))
	linePoints = [];
	canvasPositionVecs = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
	tempRvec = None
	oldFt = 0
	k, dist = calibration.load_coefficients('calibration_charuco.yml')
	linePoints.append([[[-16],[0],[95]],[0,255,0]])
	linePoints.append([[[-16],[0],[70]],[0,255,0]])
	linePoints.append([[[-16],[10],[70]],[0,255,0]])
	linePoints.append([[[-24],[10],[70]],[0,255,0]])
	while got_image:
		ft = time.time()
		thresh_img = convertToBinary(img)
		# create aruco board & dict
		aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
		arucoParams = cv2.aruco.DetectorParameters_create()
		#detect markers
		corners, ids, rejected = cv2.aruco.detectMarkers(thresh_img, aruco_dict, parameters=arucoParams, cameraMatrix=k)
		canvasVecs = [None, None, None, None]
		markerPoint = None

		for i in range(4):
			canvasVecs[i] = None
			SQUARE_LENGTH = 2.96 #2.778125
			MARKER_LENGTH = SQUARE_LENGTH * (5.0 / 7.0)
			board = getCharucoBoard(5, 7, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict, i)
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
						#
						img = drawPyramid(img, k, rvec, tvec, (255,0,0))
						markerPoint = tvec;
						#linePoints.append([tvec, (0,255,0)])
					else:
						img = drawPyramid(img, k, rvec, tvec)
						canvasVecs[i] = [tvec, rvec]
						if i==1 and frame == 0:
							tempRvec = rvec
		if markerPoint is None:
			linePoints.append(None)
		else:
			#calculate canvas position
			if(calculateCanvasLocation(img, k, canvasVecs, canvasPositionVecs) is not None):
				canvasLoc, canvasRot = calculateCanvasLocation(img, k, canvasVecs, canvasPositionVecs)
				if pointIsValid(img, k, markerPoint):
					print(canvasLoc,"\n\n", markerPoint, "done")
					linePoints.append([markerPoint-canvasLoc, (0,255,0)])
				if(tempRvec is not None):
					img = drawLines(img, linePoints, canvasLoc, k, canvasRot)

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

# eventually will use multiple canvas boards and their positions
# and then average them to get a more stable canvas location
# right now it just returns the position of charuco board id == 1
def calculateCanvasLocation(img, k, canvasVecs, canvasPositionVecs):
	if pointIsValid(img, k, canvasVecs[1]):
		return canvasVecs[1]
	else:
		return None


def pointIsValid(img, k, point):
	if point is None:
		return False
	if np.linalg.norm(point) == 0.0: #or isOffScreen(img, point, k)
		return False
	return True


def isOffScreen(img, point, k):
	axes = np.matrix([[0, 0, 0]])
	pts, jacobian = cv2.projectPoints(np.float32(axes), np.array([[0.0], [0.0], [0.0]]), point, k, None)
	s = img.shape
	pts = pts[0][0]
	return pts[0] > s[0] or pts[1] > s[1] or pts[0] < 0 or pts[1] < 0


def drawLines(img, pointsArr, canvasLoc, k, rvec):
	axes = np.matrix([[0,0,0]])

	for i in range(len(pointsArr) - 1):
		#check if there is a gap, if there is, skip to next non gapped

		if pointsArr[i] is not None and pointsArr[i + 1] is not None:
			projPoint0 = np.matrix(pointsArr[i][0])
			projPoint1 = np.matrix(pointsArr[i+1][0])
			# imagePoints0, jacobian = cv2.projectPoints(np.float32(axes), rvec, pointsArr[i][0]+canvasLoc, k, None)
			# imagePoints1, jacobian = cv2.projectPoints(np.float32(axes), rvec, pointsArr[i + 1][0]+canvasLoc, k, None)
			imagePoints0, jacobian = cv2.projectPoints(np.float32(projPoint0), rvec, canvasLoc, k, None)
			imagePoints1, jacobian = cv2.projectPoints(np.float32(projPoint1), rvec, canvasLoc, k, None)
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





