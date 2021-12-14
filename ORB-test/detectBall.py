import cv2
import numpy as np
def detect_features(bgr_img, show_features=False):
    detector = cv2.ORB_create()

    # Extract keypoints and descriptors from image.
    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, mask=None)

    # Optionally draw detected keypoints.
    if show_features:
        # Possible flags: DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, DRAW_MATCHES_FLAGS_DEFAULT
        bgr_display = bgr_img.copy()
        cv2.drawKeypoints(image=bgr_display, keypoints=keypoints,
                          outImage=bgr_display,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Features", bgr_display)
        print("Number of keypoints: ", len(keypoints))
        cv2.waitKey(0)

    return keypoints, descriptors
def rotate_and_scale(vt, scale_factor, angle_diff):
    theta = np.radians(angle_diff)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    vq = R @ vt
    vq = vq * scale_factor
    return vq
def find_cluster(query_img, keypoints_query, train_img, keypoints_train, matches,
                 show_votes=False):
    hq = query_img.shape[0]
    wq = query_img.shape[1]

    max_scale = 4.0  # Scale differences go from 0 to max_scale

    # Our accumulator array is a 4D array of empty lists. These are the number of bins
    # for each of the dimensions.
    num_bins_height = 5
    num_bins_width = 5
    num_bins_scale = 5
    num_bins_ang = 8

    # It is easier to have a 1 dimensional array instead of a 4 dimensional array.
    # Just convert subscripts (h,w,s,a) to indices idx.
    size_acc = num_bins_height * num_bins_width * num_bins_scale * num_bins_ang
    acc_array = [[] for idx in range(size_acc)]

    ht = train_img.shape[0]
    wt = train_img.shape[1]

    # Vote into accumulator array.
    for match in matches:
        qi = match.queryIdx  # Index of query keypoint
        ti = match.trainIdx  # Index of training keypoint that matched

        # Get data for training image.
        kp_train = keypoints_train[ti]
        at = kp_train.angle
        st = kp_train.size
        pt = np.array(kp_train.pt)  # training keypoint location
        mt = np.array([wt / 2, ht / 2])  # Center of training image
        vt = mt - pt  # Vector from keypoint to center

        # Get data for query image.
        kp_query = keypoints_query[qi]
        aq = kp_query.angle
        sq = kp_query.size
        pq = np.array(kp_query.pt)

        # Rotate and scale the vector to the marker point.
        scale_factor = sq / st
        angle_diff = aq - at
        angle_diff = (angle_diff + 360) % 360  # Force angle to between 0..360 degrees
        vq = rotate_and_scale(vt, scale_factor, angle_diff)
        mq = pq + vq

        if show_votes:
            # print("Scale diff %f, angle diff %f" % (scale_factor, angle_diff))

            # Display training image.
            train_img_display = train_img.copy()
            cv2.drawKeypoints(image=train_img_display, keypoints=[kp_train],
                              outImage=train_img_display,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawMarker(img=train_img_display, position=(int(mt[0]), int(mt[1])),
                           color=(255, 0, 0),
                           markerType=cv2.MARKER_DIAMOND)
            cv2.line(img=train_img_display,
                     pt1=(int(pt[0]), int(pt[1])), pt2=(int(mt[0]), int(mt[1])),
                     color=(255, 0, 0), thickness=2)
            cv2.imshow("Training keypoint", train_img_display)

            # Display query image.
            query_img_display = query_img.copy()
            cv2.drawKeypoints(image=query_img_display, keypoints=[kp_query],
                              outImage=query_img_display,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.line(img=query_img_display,
                     pt1=(int(pq[0]), int(pq[1])), pt2=(int(mq[0]), int(mq[1])),
                     color=(255, 0, 0), thickness=2)
            cv2.imshow("Query keypoint", query_img_display)
            cv2.waitKey(100)

        # Compute the cell of the accumulator array, that this match should be stored in.
        row_subscript = int(round(num_bins_height * (mq[1] / hq)))
        col_subscript = int(round(num_bins_width * (mq[0] / wq)))
        if row_subscript >= 0 and row_subscript < num_bins_height and col_subscript >= 0 and col_subscript < num_bins_width:
            scale_subscript = int(num_bins_scale * (scale_factor / max_scale))
            if scale_subscript > num_bins_scale:
                scale_subscript = num_bins_scale - 1

            ang_subscript = int(num_bins_ang * (angle_diff / 360))
            # print(row_subscript,col_subscript, scale_subscript, ang_subscript)

            # Note: the numpy functions ravel_multi_index(), and unravel_index() convert
            # subscripts to indices, and vice versa.
            idx = np.ravel_multi_index(
                (row_subscript, col_subscript, scale_subscript, ang_subscript),
                (num_bins_height, num_bins_width, num_bins_scale, num_bins_ang))

            acc_array[idx].append(match)

    # Count matches in each bin.
    counts = [len(acc_array[idx]) for idx in range(size_acc)]

    # Find the bin with maximum number of counts.
    idx_max = np.argmax(np.array(counts))

    # Return the matches in the largest bin.
    return acc_array[idx_max]

def calc_affine_transformation(matches_in_cluster, kp_train, kp_query):
    if len(matches_in_cluster) < 3:
        # Not enough matches to calculate affine transformation.
        return None, None

    # Estimate affine transformation from training to query image points.
    # Use the "least median of squares" method for robustness. It also detects outliers.
    # Outliers are those points that have a large error relative to the median of errors.
    src_pts = np.float32([kp_train[m.trainIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    dst_pts = np.float32([kp_query[m.queryIdx].pt for m in matches_in_cluster]).reshape(
        -1, 1, 2)
    A_train_query, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC)

    return A_train_query, inliers

def detect_object_center(train_path, train_center_pt, query_path):

    bgr_train = cv2.imread(train_path)  # Get training image
    bgr_query = cv2.imread(query_path)  # Get query image

    # Extract keypoints and descriptors.
    kp_train, desc_train = detect_features(bgr_train, show_features=False)
    kp_query, desc_query = detect_features(bgr_query, show_features=False)

    matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)

    # Match query image descriptors to the training image.
    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc_query, desc_train, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good

    matches = find_cluster(bgr_query, kp_query, bgr_train, kp_train, matches,
                            show_votes=False)

    # Calculate an affine transformation from the training image to the query image.
    A_train_query, inliers = calc_affine_transformation(matches, kp_train, kp_query)

    # Transform training center with A matrix to get query center point
    query_center_pt = A_train_query @ train_center_pt

    return query_center_pt
