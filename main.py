import time

import cv2
import numpy as np
from calibration import calibration


def getCharucoBoard(x_dim, y_dim, square_size, marker_size, aruco_dict, offset):
	board = cv2.aruco.CharucoBoard_create(x_dim,y_dim,square_size,marker_size,aruco_dict)
	board.ids += offset*int((x_dim*y_dim)/2.0);
	return board





def main(flags):
	frame = 0 #reset frame count
	#get first frame
	video_capture = cv2.VideoCapture("videos/demos/flower.mp4");
	got_image, img = video_capture.read()
	img_height, img_width = img.shape[0], img.shape[1]
	if(flags['writeVideo']):
		# start video writer
		fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
		videoWriter = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=30.0,
									  frameSize=(img_width, img_height))
	linePoints = [];
	canvasPositionVecs = [np.array([[0.0], [0.0], [0.0]])]*4
	oldFt = 0
	k, dist = calibration.load_coefficients('calibration/calibration_charuco.yml')
	# create aruco board & dict
	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
	arucoParams = cv2.aruco.DetectorParameters_create()
	LINE_COLOR = flags["color"]

	if(flags["GENERATE_IMAGES"]):
		# TEST CODE TO GENERATE 5 DIFFERENT CHARUCO BOARDS
		for i in range(4):
			board = getCharucoBoard(5, 7, 10, 8, aruco_dict, i)
			boardImg = board.draw((2000,2000))
			cv2.imwrite("testLarge"+str(i)+".png", boardImg)
		cv2.waitKey(0)
	while got_image:
		ft = time.time()
		thresh_img = convertToBinary(img)
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
			#skip to next board if current one is invalid
			if ids is None: continue
			resp, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
				newCorners, newIds, thresh_img, board, cameraMatrix=k
			)
			# skip to next board if current one is invalid
			if charucoIds is None: continue

			if(flags["showDebug"]):
				img = cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, (0, 0, 255))
			_, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
				charucoCorners, charucoIds, board,
				cameraMatrix=k, distCoeffs=None,
				rvec=np.array([[0.0], [0.0], [0.0]]),
				tvec=np.array([[0.0], [0.0], [0.0]]))
			#check if the current board is the marker
			if i == flags['MARKER_ID']:
				if(flags["showDebug"]):
					img = drawPyramid(img, k, rvec, tvec, (255,0,0))
				markerPoint = tvec;
				#linePoints.append([tvec, (0,255,0)])
			else:
				if (flags["showDebug"]):
					img = drawPyramid(img, k, rvec, tvec, (i*63,i*63,i*63))
				canvasVecs[i] = [tvec, rvec]

		if markerPoint is None:
			linePoints.append(None)
		else:
			#calculate canvas position
			if(calculateCanvasLocation(img, k, canvasVecs, canvasPositionVecs) is not None):
				canvasLoc, canvasRot = calculateCanvasLocation(img, k, canvasVecs, canvasPositionVecs)
				if pointsAreValid(img, k, [markerPoint, canvasLoc]):
					R, _ = cv2.Rodrigues(canvasRot*-1)
					rotated = R @ (markerPoint-canvasLoc)
					linePoints.append([rotated, LINE_COLOR])
		if (calculateCanvasLocation(img, k, canvasVecs, canvasPositionVecs) is not None):
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
# right now we just average the rotations because rotation is far more unstable than position.
# we just use position of canvas ID 1
def calculateCanvasLocation(img, k, canvasVecs, canvasPositionVecs):
	out = [np.array([[0.0], [0.0], [0.0]]), np.array([[0.0], [0.0], [0.0]])]
	count = 0;
	for i in range(4):
		#skip canvas if it isn't there or isn't valid
		if canvasVecs[i] is None or not pointIsValid(img, k, canvasVecs[1]):
			continue
		# out[0] += canvasVecs[i][0]-canvasPositionVecs[i]
		# calculate the delta between the other canvases to see if there is an error
		diff = 0
		for j in range(4):
			if i != j and canvasVecs[j] is not None:
				diff += np.linalg.norm(canvasVecs[i][1]-canvasVecs[j][1])
		#if there is an error, we disguard the bad value from the average
		if(diff < .5):
			out[1] += canvasVecs[i][1]
			count += 1
	if count == 0 or canvasVecs[3] is None:
		return None
	return [canvasVecs[3][0], out[1] / count]


def pointIsValid(img, k, point):
	if point is None or np.linalg.norm(point) == 0.0: #or isOffScreen(img, point, k)
		return False
	return True

def pointsAreValid(img, k, pArr):
	for p in pArr:
		if not pointIsValid(img, k, p):
			return False
	return True

def isOffScreen(img, point, k):
	axes = np.matrix([[0, 0, 0]])
	pts, jacobian = cv2.projectPoints(np.float32(axes), np.array([[0.0], [0.0], [0.0]]), point, k, None)
	s = img.shape
	pts = pts[0][0]
	return pts[0] > s[0] or pts[1] > s[1] or pts[0] < 0 or pts[1] < 0


def drawLines(img, pointsArr, canvasLoc, k, canvasRvec):
	axes = np.matrix([[0,0,0]])
	for i in range(len(pointsArr) - 1):
		#check if there is a gap, if there is, skip to next non gapped

		if pointsArr[i] is not None and pointsArr[i + 1] is not None:
			projPoint0 = np.matrix(pointsArr[i][0])
			projPoint1 = np.matrix(pointsArr[i+1][0])
			imagePoints0, jacobian = cv2.projectPoints(np.float32(projPoint0), canvasRvec, canvasLoc, k, None)
			imagePoints1, jacobian = cv2.projectPoints(np.float32(projPoint1), canvasRvec, canvasLoc, k, None)
			# imagePoints0, jacobian = cv2.projectPoints(np.float32(axes), canvasRvec, pointsArr[i][0] + canvasLoc, k, None)
			# imagePoints1, jacobian = cv2.projectPoints(np.float32(axes), canvasRvec, pointsArr[i + 1][0] + canvasLoc, k, None)
			img = cv2.line(img,
						   tuple(np.int32(imagePoints0).ravel()),
						   tuple(np.int32(imagePoints1).ravel()), pointsArr[i][1], 5)



	return img

def drawPyramid(img, k, rvec, tvec, color=(0,0,255)):
	scale = 2;
	pHeight = 10;
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
		C=0
	)
	return thresh_img


if __name__ == "__main__":
	main({
		'writeVideo' : True,
		'showBinary' : False,
		'showFPS' : False,
		'showDebug' : True,
		'color' : (0, 255, 0),
		'MARKER_ID' : 0,
		'GENERATE_IMAGES' : False
	})





