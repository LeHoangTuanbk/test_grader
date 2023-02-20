import cv2
import numpy as np 
import process
import imutils
##
path = "3.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1,2,3,1,4]
font = cv2.FONT_HERSHEY_COMPLEX
#### preprocessing
img = cv2.imread(path)
img = cv2.resize(img,(widthImg,heightImg))
imgContours = img.copy()
imgContours00 = img.copy()
imgSoft = img.copy()
imgBigContour = img.copy()
imgFinal = img.copy()



imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgGray01 = imgGray.copy()
_, threshold = cv2.threshold(imgGray, 240, 255, cv2.THRESH_BINARY_INV) # tham so nhị phân hóa ổn.


contours, hier = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # tim khong bao gom trong

rectCountour = process.rectCountour(contours)
#cv2.drawContours(imgContours,contours,-1,(0,255,0),1)
# ham sap xep cac contours dang loi
# rectCountour,_ = process.sort_contours(contours,method="left-to-right")
# rectCountour,_ = process.sort_contours(contours,method="top-to-bottom")


cv2.drawContours(imgSoft,rectCountour[0],-1,(0,255,0),1)
cv2.imshow("Anh add contours ",imgSoft)


# for i in rectCountour:
#   y=1
#   cv2.drawContours(imgSoft,i,-1,(0,255,0),1)
#   cv2.imshow("Anh soft contours"+str(y),imgSoft)
#   y+=1
#   print(y)

# for i in SoftedContours:
#   cv2.imshow(i,i)


# cv2.imshow("imgGray", imgGray01)
# cv2.imshow("threshold",threshold)
# cv2.imshow("Hien thi test", imgContours)
# cv2.imshow("Contours HCN", imgContours00)


cv2.waitKey(0)
cv2.destroyAllWindows()
