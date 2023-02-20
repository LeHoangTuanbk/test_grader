import cv2
import numpy as np 

widthImg = 500
heightImg = int(500*29.4/21)

def rectCountour(coutours):
  rectCon = []
  for i in coutours:
    area= cv2.contourArea(i)
    #print("Area",area)
    if area>50000:
      peri = cv2.arcLength(i,True)
      approx = cv2.approxPolyDP(i,0.02*peri,True)
      #print("Corner Points",len(approx))
      if len(approx) == 4:
        rectCon.append(i) 
  # sap xep theo dien tich lon => nho
  #rectCon = sorted(rectCon,key= cv2.contourArea,reverse = True)
  return rectCon
def getContours_Need(rectCountour,img):
  #Lay tam cua cac contours va tach ra tung phan SBD, MD, cau 1-25, 25-50
  toaDoTam = []
  size = img.shape
  h = size[0]
  w = size[1]
  #print(size)
  SBDorMD = []
  for c in rectCountour:
    # Doc chieu Ox, Ngang chieu Oy
    M = cv2.moments(c)
    cX = int(M["m01"] / M["m00"])
    cY = int(M["m10"] / M["m00"])
    #toaDoTam.append([cX,cY])
    if cX > h/2 :
      if cY < w/2:
        Cau1_25 = c
      else: 
        Cau26_50 = c
    elif cY > w/2:
      SBDorMD.append(c)
  Y_SBDorMD = []
  for c in SBDorMD:
    M = cv2.moments(c)
    cY = int(M["m10"] / M["m00"])
    Y_SBDorMD.append(cY)
  if Y_SBDorMD[0] < Y_SBDorMD[1]:
    SBD = SBDorMD[0]
    MD = SBDorMD[1]
  else: 
    SBD = SBDorMD[1]
    MD = SBDorMD[0]
  return (Cau1_25,Cau26_50,SBD,MD)
def getCornerPoints(cont):
  peri = cv2.arcLength(cont,True)
  approx = cv2.approxPolyDP(cont,0.02*peri,True)
  return approx
def reorder(myPoints):
  myPoints = myPoints.reshape((4,2))
  myPointsNew = np.zeros((4,1,2),np.int32)
  add = myPoints.sum(1)
  #print(myPoints)
  #print(add)
  myPointsNew[0]= myPoints[np.argmin(add)] # [0,0]
  myPointsNew[3]= myPoints[np.argmax(add)] # [w,h]
  diff = np.diff(myPoints,axis=1)
  myPointsNew[1]= myPoints[np.argmin(diff)] # [w,0]
  myPointsNew[2]= myPoints[np.argmax(diff)] # [0,h]
  #print(diff)
  #print(np.argmax(diff),np.argmin(diff))
  return myPointsNew 
def birdView(contours, img_bird):
  pt1 = np.float32(contours)
  pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
  matrix = cv2.getPerspectiveTransform(pt1,pt2)
  contours = cv2.warpPerspective(img_bird,matrix,(widthImg,heightImg))
  return contours
def splitBoxesSBD(img,x,y):
  # x: chieu ngan, y: chieu doc
  #print(img.shape)
  cols = np.hsplit(img,x) # cat theo chieu doc, theo cot
  boxes = []
  for r in cols: 
    rows = np.vsplit(r,y) # cat theo chieu ngang, theo hang
    for box in rows: 
      boxes.append(box)
      # cv2.imshow("split",box)
      # cv2.waitKey(0)
  #cv2.destroyAllWindows()
  return boxes
def splitBoxesMD(img,x,y):
  # x: chieu ngan, y: chieu doc
  #print(img.shape)
  cols = np.hsplit(img,x) # cat theo chieu doc, theo cot
  boxes = []
  for r in cols: 
    rows = np.vsplit(r,y) # cat theo chieu ngang, theo hang
    for box in rows: 
      boxes.append(box)
      # cv2.imshow("split",box)
      # cv2.waitKey(0)
  #cv2.destroyAllWindows()
  return boxes
def GetSBD(Box_SBD,SBD_cols,SBD_rows):
  # tao ma tran 6 hang, 10 cot cho sbd
  myPixelVal_SBD = np.zeros((SBD_cols,SBD_rows))
  countC=0
  countR=0
  for image in Box_SBD:
    totalPixelsinImage = cv2.countNonZero(image)
    myPixelVal_SBD[countR][countC] = totalPixelsinImage
    countC +=1
    if (countC == SBD_rows): 
      countR+=1
      countC=0
  #print(myPixelVal_SBD)
  myIndex = []
  for x in range (0,SBD_cols):
    arr = myPixelVal_SBD[x]
    myIndexVal = np.where(arr == np.amax(arr))
    myIndex.append(myIndexVal[0][0])
  #print(myIndex)
  return myIndex
def GetMD(Box_MD0,MD_cols,MD_rows):
  myPixelVal_MD = np.zeros((MD_cols,MD_rows))
  countC=0
  countR=0
  for image in Box_MD0:
    totalPixelsinImage = cv2.countNonZero(image)
    myPixelVal_MD[countR][countC] = totalPixelsinImage
    countC +=1
    if (countC == MD_rows): 
      countR+=1
      countC=0
  myIndex = []
  for x in range (0,MD_cols):
    arr = myPixelVal_MD[x]
    myIndexVal = np.where(arr == np.amax(arr))
    myIndex.append(myIndexVal[0][0])
  #print(myIndex)
  return myIndex
def splitBoxesCau1_25(thresholdCau1_25,Cau1_25_cols,Cau1_25_rows): # nen xoa trong tuong lai
  blockCau = np.vsplit(thresholdCau1_25,5)
  for block in blockCau:
    print(block.shape)
    Giamx= int(block.shape[0]/32+10)
    xSau = int(31*block.shape[0]/32)
    Giamy= int(block.shape[1]/20-20)
    ySau= int(19*block.shape[1]/20)
    block = block[Giamx:xSau, Giamy:ySau, :]
    block = cv2.resize(block,(450,235))
    print(block.shape)
    blockCauNho = np.vsplit(block,5)
    boxes = []
    # a= "gray_image"
    # #cv2.imwrite("Anh%s.jpg"%i, block)
    # #print(i)
    # 
    # cv2.imshow("1-5", block)
    # cv2.waitKey(0)
    for boxBig in blockCauNho:
      # cv2.imshow("split44", boxBig)
      # cv2.waitKey(0)
      DapAn = np.hsplit(boxBig,Cau1_25_cols)
      i = 0
      for box in DapAn:
        if(i%5 != 0):
          boxes.append(box)
        i+=1
        # cv2.imshow("split",box)
        # cv2.waitKey(0)
    for bo in box:
      cv2.imshow("bo ", bo )
      cv2.waitKey(0)

  cv2.destroyAllWindows()
  return boxes
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
def showMDandSBD(img,cols,rows,getList):
  # print(img.shape)
  # print(getList)
  secW =int(img.shape[1]/cols)
  secH =int(img.shape[0]/rows)
  for x in range(0,len(getList)):
    # x chieu doc, y chieu ngang
    cX = x*secW+secW//2
    cY = getList[x]*secH+secH//2
    cv2.circle(img,(cX,cY),20,(0,255,0),cv2.FILLED) # ham draw: cX: ngang, cY: chieu doc

  return img
  


  # secW = int(img.shape[1]/questions)
  # secH = int(img.shape[0]/choices)
  # for x in range(0,questions):
  #   myAns = myIndex[x]
  #   cX = (myAns*secW)+secW//2
  #   cY = (x*secH)+secH//2
  #   if grading[x] == 1:
  #     myColor = (0,255,0)
  #   else:
  #     myColor= (0,0,255)
  #     correctAns = ans[x]
  #     cv2.circle(img,((correctAns*secW)+secW//2,(x*secH)+secH//2),30,(0,255,0),cv2.FILLED)
  #   cv2.circle(img,(cX,cY),60,myColor,cv2.FILLED)
  
  
  
  
  
  

  

  
  
  
  # cols = np.hsplit(img,x) # cat theo chieu doc, theo cot
  # boxes = []
  # for r in cols: 
  #   rows = np.vsplit(r,y) # cat theo chieu ngang, theo hang
  #   for box in rows: 
  #     boxes.append(box)
  #     # cv2.imshow("split",box)
  #     # cv2.waitKey(0)
  # #cv2.destroyAllWindows()
  # return boxes
