import cv2
import numpy as np
img1=cv2.imread("Narsimha1.jpg",cv2.IMREAD_GRAYSCALE)
cap=cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()
kp_img,des_img=sift.detectAndCompute(img1,None)
#feature detection
index_params=dict(algorithm=0,trees=5)
search_params=dict()
flann=cv2.FlannBasedMatcher(index_params,search_params)
while True:
    ret, frame = cap.read()
    grayvideo=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kp_video, des_video = sift.detectAndCompute(grayvideo, None)
    matches=flann.knnMatch(des_img,des_video,k=2)
    good=[]
    for m,n in matches:
        if m.distance<n.distance*0.6:
            good.append(m)
    img=cv2.drawMatches(img1,kp_img,grayvideo,kp_video,good,grayvideo)


    if len(good)>10:
        query_pt=np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        train_pt = np.float32([kp_video[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        matrix,mask=cv2.findHomography(query_pt,train_pt,cv2.RANSAC,5.0)
        matches_mask=mask.ravel().tolist()
        h,w=img1.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 3)
        cv2.imshow("homography",homography)
    else:
        cv2.imshow("homography", frame)

    # cv2.imshow("view", grayvideo)
    # cv2.imshow("img1", img1)
    # cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == "s":
        break
cap.release()
cv2.destroyAllWindows()
