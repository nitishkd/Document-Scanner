import cv2
import numpy as np
import os
from PIL import Image


image = cv2.imread("notes.jpg")

ratio = image.shape[0] / 800.0
#read the imagea and resize it.
(h, w) = image.shape[:2]
r = 800 / float(h) 
width = int(w * r)
dim = None

res = cv2.resize(image,(width,800), interpolation = cv2.INTER_AREA)
#cv2.imshow("image",res)
print(dim)

print ratio
#convert into grayscale, blur it and find its edges
orig = image.copy()
gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
#cv2.imshow("gaussian blur",blur)

#use Canny edge detection method for detecting edges
edged = cv2.Canny(blur,75,200)
#cv2.imshow("edged",edged)

cv2.waitKey(0)
cv2.destroyAllWindows()

#finding the largest contours in edged image, keeping the largest one and
#initialize the screen contour

(cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]

for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.02*peri,True)
    if len(approx) == 4:
         screenCnt = approx
         break

print("finding contours of paper")
cv2.drawContours(res,[screenCnt],-1,(0,0,255),2)
#cv2.imshow("Selected Border",res)

pts = screenCnt.reshape(4, 2) * ratio
rect = np.zeros((4, 2), dtype = "float32")

s = pts.sum(axis = 1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

(tl,tr,br,bl) = rect
print ((tl,tr,br,bl))

#new dimensions of image

widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

X = cv2.getPerspectiveTransform(rect, dst)
#cv2.imshow("original1",orig)
crop = cv2.warpPerspective(orig, X, (maxWidth, maxHeight))
crop = crop.astype("uint8") * 255


crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)

ret, im_th1 = cv2.threshold(crop,115,255,cv2.THRESH_BINARY_INV)
ret, im_th2 = cv2.threshold(crop,115,255,cv2.THRESH_BINARY)
im_th3 = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)

im_th2 = cv2.resize(im_th2,(700,800),interpolation = cv2.INTER_AREA)
im_th1 = cv2.resize(im_th1,(700,800),interpolation = cv2.INTER_AREA)
im_th3 = cv2.resize(im_th3,(700,800),interpolation = cv2.INTER_AREA)


cv2.imshow("Final Image 1",im_th1)
cv2.imshow("Final Image 2",im_th2)
#cv2.imshow("adaptive",im_th3)
cv2.imwrite("1.jpg",im_th1)
cv2.imwrite("2.jpg",im_th2)
im = Image.open("1.jpg")
im.save("Scanned_doc.pdf","PDF", Quality = 100)


cv2.waitKey(0)
cv2.destroyAllWindows()
