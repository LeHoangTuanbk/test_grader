import cv2
import numpy as np 
import process
import imutils

####################################
path = "5AnhCheoTrai.jpg"
widthImg = 500
heightImg = int(500*29.4/21)
questions = 50
# Dap an de nguoi ta sai cau 24, 49,50
DapAn = [[1, 3], [2, 2], [3, 2], [4, 1], [5, 1], [6, 2], [7, 0], [8, 1], [9, 2], [10, 3], 
[11, 0], [12, 2], [13, 0], [14, 3], [15, 1], [16, 2], [17, 1], [18, 3], [19, 0], [20, 2], 
[21, 0], [22, 2], [23, 1], [24, 1], [25, 0],[26, 3], [27, 2], [28, 0], [29, 1], [30, 2],
[31, 2], [32, 1], [33, 3], [34, 0], [35, 3],[36, 1], [37, 2], [38, 0], [39, 3], [40, 0], 
[41, 3], [42, 2], [43, 1], [44, 2], [45, 0], [46, 0],[47, 0], [48, 1], [49, 1], [50, 1]]
MDThuNhat = "123"
font = cv2.FONT_HERSHEY_COMPLEX


###################################### Buoc 1 ##############################
img = cv2.imread(path)
# img = cv2.resize(img,(widthImg,heightImg))
img_bird = img.copy()
imgContours = img.copy()
imgContours00 = img.copy()
imgSoft = img.copy()
imgPhieuTraLoi02 = img.copy()
imgPhieuTraLoi01 = img.copy()
imgMaDe = img.copy()
imgSBD = img.copy()
imgFinal = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgGray01 = imgGray.copy()
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
_, threshold = cv2.threshold(imgBlur, 230, 255, cv2.THRESH_BINARY_INV) # tham so nhị phân hóa ổn.
imgCanny = cv2.Canny(threshold,5,50)

contours, hierarchy = cv2.findContours(threshold ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # tim khong bao gom trong
rectCountour = process.rectCountour(contours)
#imgCon = cv2.drawContours(imgSoft,rectCountour,-1,(0,255,0),10)
Cau1_25,Cau26_50,SBD,MD = process.getContours_Need(rectCountour,img)

cv2.drawContours(imgContours, Cau1_25, -1, (0, 255, 0), 5)
cv2.drawContours(imgContours, Cau26_50, -1, (0, 255, 0), 5)
cv2.drawContours(imgContours, SBD, -1, (0, 255, 0), 5)
cv2.drawContours(imgContours, MD, -1, (0, 255, 0), 5)
cv2.imshow("B1:Anh lay duoc SBD,MD,1-50",cv2.resize(imgContours,(500,700)))
cv2.waitKey(0)

################################Buoc 2#################################
# lay cac dinh cua contours

Cau1_25 = process.getCornerPoints(Cau1_25)
Cau1_25 = process.reorder(Cau1_25)
Cau26_50 = process.getCornerPoints(Cau26_50)
Cau26_50 = process.reorder(Cau26_50)
SBD = process.getCornerPoints(SBD)
SBD = process.reorder(SBD)
SBD_toaDo = SBD
MD = process.getCornerPoints(MD)
MD = process.reorder(MD)

#hien thi ra birdview
Cau1_25 = process.birdView(Cau1_25, img_bird)
Cau26_50 = process.birdView(Cau26_50, img_bird)
SBD = process.birdView(SBD, img_bird)
#print(SBD.shape)
MD = process.birdView(MD, img_bird)

cv2.imshow("B2:Cau1_25",Cau1_25)
cv2.imwrite("B2:Cau1_25.jpg", Cau1_25)
cv2.imshow("B2:Cau26_50  ",Cau26_50)
cv2.imshow("B2:SBD  ",SBD)
cv2.imshow("B2:MD ",MD)
cv2.waitKey(0)


######################### B3: lay ma de, SBD ################################
SBD_rows = 10
SBD_cols = 6
SBD0 = cv2.resize(SBD,(600,1000))
#print(SBD0.shape)
SBD0 = cv2.cvtColor(SBD0,cv2.COLOR_BGR2GRAY)
_, thresholdSBD = cv2.threshold(SBD0, 230, 255, cv2.THRESH_BINARY_INV) # nguong 230, de 4 dung, nguong 200, de 5 dung
Box_SBD = process.splitBoxesSBD(thresholdSBD,SBD_cols,SBD_rows)

SBD_list = process.GetSBD(Box_SBD,SBD_cols,SBD_rows)
SBD_String = ""
for i in SBD_list:
	a = str(i)
	#print(a)
	SBD_String += a
print("So bao danh:",SBD_String)

# lay ma de cua thi sinh
MD_rows = 10
MD_cols = 3
MD0 = cv2.resize(MD,(300,1000))
#print(MD0.shape)
MD0 = cv2.cvtColor(MD0,cv2.COLOR_BGR2GRAY)
_, thresholdMD = cv2.threshold(MD0, 230, 255, cv2.THRESH_BINARY_INV)
Box_MD0 = process.splitBoxesMD(thresholdMD,MD_cols,MD_rows)

MD_list = process.GetMD(Box_MD0,MD_cols,MD_rows)
#print("Ma de",MD_list)
MD_String = ""
for i in MD_list:
	a = str(i)
	#print(a)
	MD_String += a
print("Ma de:",MD_String)
#Phan hien thi danh dau SBD va MD
#
widthImgFinal = imgFinal.shape[1]
heightImgFinal = imgFinal.shape[0]
imRawDrawing = SBD.copy()
inRawDrawingSBD = process.showMDandSBD(imRawDrawing,6,10,SBD_list)

imRawDrawing = MD.copy()
inRawDrawingMD = process.showMDandSBD(imRawDrawing,3,10,MD_list)
cv2.imshow("B3:SBDNhanDien",inRawDrawingSBD)
cv2.imshow("B3:MDNhanDien",inRawDrawingMD)
cv2.waitKey(0)





# ###############B4: Lay dap an cua thi sinh #############################

###########Cau 1_25


image = Cau1_25
image0 = image.copy()
image1 = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
_, threshold = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)
cnts, hier = cv2.findContours(threshold, cv2.RETR_LIST,
	cv2.CHAIN_APPROX_SIMPLE)
#print(cnts)
image0 = cv2.drawContours(image0, cnts, -1, (0, 255, 0), 2)
# cv2.imshow("test1_25",threshold)
# cv2.waitKey(0)
#print(image0.shape)
w = image0.shape[1]/5

cnts0 = []
for i in cnts:
  area= cv2.contourArea(i)
  #print("Area",area)
  if area > 300 and area < 10000:
    M = cv2.moments(i)
    cX = int(M["m01"] / M["m00"])
    cY = int(M["m10"] / M["m00"])
    if cY > w:
      cnts0.append(i)
      #cv2.drawContours(image1, [i], -1, (0, 255, 0), 2)

#image1 = cv2.drawContours(image1, cnts0, -1, (0, 255, 0), 2)

# Sap xep cac contour theo hang
cnts0 = process.sort_contours(cnts0, method="top-to-bottom")[0]
cnts1 = []
# cnts1[0:3] = process.sort_contours(cnts0[0:4],method="left-to-right")[0]

a = len(cnts0)
# sap xep contour theo chieu trai sang phai
for i in range(0,len(cnts0)):
	if (i%4) == 0:
		#print(i)
		cnts1[i:i+3] = process.sort_contours(cnts0[i:i+4],method="left-to-right")[0]
countC = 0
countR = 0
myPixelVal_1_25 = np.zeros((25,4))
for i in cnts1:
	mask = np.zeros(threshold.shape, dtype="uint8")
	cv2.drawContours(mask, [i], -1, 255, -1)
	mask = cv2.bitwise_and(threshold, threshold, mask=mask)
	totalPixelsinImage = cv2.countNonZero(mask)
	#print(totalPixelsinImage)
	myPixelVal_1_25[countR][countC] = totalPixelsinImage
	countC +=1
	if (countC == 4):
		countR+=1
		countC=0
#print(myPixelVal_1_25)
#Lay duoc o thi sinh to va so sanh voi dap an
myIndex = []
for x in range (0,25):
  arr = myPixelVal_1_25[x]
  myIndexVal = np.where(arr == np.amax(arr))
  myIndex.append([x+1,myIndexVal[0][0]])
#print("cau1_25 thi sinh khoanh: ",myIndex)
	
#to de hien thi dap an 

wDapAn_1_25 = []
for i in range(25):
	for j in range(4):
		if j == DapAn[i][1]:
			wDapAn_1_25.append(1)
		else:
			wDapAn_1_25.append(0)
#print("list to",wDapAn_1_25)
if MD_String == MDThuNhat:
	#print("okay")
	j=0
	for i in cnts1:
		if wDapAn_1_25[j] == 1:
			cv2.drawContours(Cau1_25, [i], -1, (0, 255, 0), 5)
		j+=1
cv2.imshow("B4:Cau1_25",Cau1_25)
#cv2.waitKey(0)

# print("Ma tran to mau",wToMau)

	


#########Cau25_50



image = Cau26_50
image0 = image.copy()
image1 = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
_, threshold = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

cnts, hier = cv2.findContours(threshold, cv2.RETR_LIST,
	cv2.CHAIN_APPROX_SIMPLE)
#print(cnts)
image0 = cv2.drawContours(image0, cnts, -1, (0, 255, 0), 2)
#cv2.imshow("test25_50",threshold)
#cv2.waitKey(0)
#print(image0.shape)
w = image0.shape[1]/5

cnts0 = []
for i in cnts:
  area= cv2.contourArea(i)
  #print("Area",area)
  if area > 300 and area < 10000:
    M = cv2.moments(i)
    cX = int(M["m01"] / M["m00"])
    cY = int(M["m10"] / M["m00"])
    if cY > w:
      cnts0.append(i)
      #cv2.drawContours(image1, [i], -1, (0, 255, 0), 2)

#image1 = cv2.drawContours(image1, cnts0, -1, (0, 255, 0), 2)

# Sap xep cac contour theo hang
cnts0 = process.sort_contours(cnts0, method="top-to-bottom")[0]
cnts1 = []
# cnts1[0:3] = process.sort_contours(cnts0[0:4],method="left-to-right")[0]

a = len(cnts0)
# sap xep contour theo chieu trai sang phai
for i in range(0,len(cnts0)):
	if (i%4) == 0:
		#print(i)
		cnts1[i:i+3] = process.sort_contours(cnts0[i:i+4],method="left-to-right")[0]
countC = 0
countR = 0
myPixelVal_1_25 = np.zeros((25,4))
for i in cnts1:
	mask = np.zeros(threshold.shape, dtype="uint8")
	cv2.drawContours(mask, [i], -1, 255, -1)
	mask = cv2.bitwise_and(threshold, threshold, mask=mask)
	totalPixelsinImage = cv2.countNonZero(mask)
	#print(totalPixelsinImage)
	myPixelVal_1_25[countR][countC] = totalPixelsinImage
	countC +=1
	if (countC == 4):
		countR+=1
		countC=0
#print(myPixelVal_1_25)
#myIndex = []
for x in range (0,25):
  arr = myPixelVal_1_25[x]
  myIndexVal = np.where(arr == np.amax(arr))
  myIndex.append([x+26,myIndexVal[0][0]])
#print("Cau25_50 thi sinh khoanh: ",myIndex)
#to de hien thi dap an 

wDapAn_26_50 = []
for i in range(25):
	for j in range(4):
		if j == DapAn[i+25][1]:
			wDapAn_26_50.append(1)
		else:
			wDapAn_26_50.append(0)
#print("list to",wDapAn_26_50)
if MD_String == MDThuNhat:
	j=0
	for i in cnts1:
		if wDapAn_26_50[j] == 1:
			cv2.drawContours(Cau26_50, [i], -1, (0, 255, 0), 5)
		j+=1
cv2.imshow("B4: Cau26_50",Cau26_50)
cv2.waitKey(0)
print("Thi sinh khoanh:\n  ",myIndex)



####################B5: cham diem va hien thi

#Cham diem cho thi sinh
z = 0
for i in range(0,50):
	if DapAn[i][1] == myIndex[i][1]:
		z+=1
print("Tong so cau dung %s/50"%z)
textIn = "SBD: "+SBD_String+" Ma de: "+MD_String+" Tong so cau dung: "+str(z)+ "/50"
cv2.putText(imgFinal,textIn,(20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,(0,0,255),2,cv2.LINE_AA)
cv2.imshow("B5:Anh dau ra",cv2.resize(imgFinal,(500,700)))
cv2.waitKey(0)





cv2.destroyAllWindows()