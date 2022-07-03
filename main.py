import cv2
import numpy as np
import imutils

image_raw = cv2.imread('images/Aglomerado.jpeg')
cv2.imshow('original', image_raw)

image = cv2.resize(image_raw, (960, 540))

#amarelo
lower_yellow = np.array([18, 98, 59], np.uint8) 
upper_yellow = np.array([36, 255, 255], np.uint8)

#verde
lower_green = np.array([44, 154, 40], np.uint8)
upper_green = np.array([67, 255, 255], np.uint8)

#vermelho
lower_red = np.array([160, 120, 60], np.uint8)  
upper_red = np.array([200, 255, 255], np.uint8)

#azul
lower_blue = np.array([86, 109, 14], np.uint8)
upper_blue = np.array([157, 255, 255], np.uint8)

#laranja
lower_orange = np.array([6, 70, 10], np.uint8)
upper_orange = np.array([20, 255, 255], np.uint8)

#marrom
lower_brown = np.array([0, 0, 0], np.uint8)
upper_brown = np.array([180, 255, 36], np.uint8)


def ContourCalculate(image):

    kernel = np.ones((5, 5), "uint8")

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening', opening)
    img_erosion = cv2.erode(opening, kernel, iterations=3)

    hsvImg = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsvImg)

    yellow_mask = cv2.inRange(hsvImg, lower_yellow, upper_yellow)
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    yellow_contours = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours = imutils.grab_contours(yellow_contours)

    green_mask = cv2.inRange(hsvImg, lower_green, upper_green)
    green_mask = cv2.dilate(green_mask, kernel)
    green_contours = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours = imutils.grab_contours(green_contours)

    red_mask = cv2.inRange(hsvImg, lower_red, upper_red)
    red_mask = cv2.dilate(red_mask,kernel)
    red_contours = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_contours = imutils.grab_contours(red_contours)

    blue_mask = cv2.inRange(hsvImg, lower_blue, upper_blue)
    blue_mask = cv2.dilate(blue_mask, kernel)
    blue_contours = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours = imutils.grab_contours(blue_contours)

    orange_mask = cv2.inRange(hsvImg, lower_orange, upper_orange)
    orange_mask = cv2.dilate(orange_mask, kernel)
    orange_contours = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    orange_contours = imutils.grab_contours(orange_contours)

    brown_mask = cv2.inRange(hsvImg, lower_brown, upper_brown)
    brown_mask = cv2.dilate(brown_mask, kernel)
    brown_contours = cv2.findContours(brown_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    brown_contours = imutils.grab_contours(brown_contours)

    cv2.imshow('mascara amarela',yellow_mask)
    cv2.imshow('mascara verde',green_mask)
    cv2.imshow('mascara vermelha',red_mask)
    cv2.imshow('mascara azul',blue_mask)
    cv2.imshow('mascara laranja',orange_mask)
    cv2.imshow('mascara marrom',brown_mask)

    lengthYellow = 0
    for y in yellow_contours:
        area1 = cv2.contourArea(y)
        if area1>1600:
            # cv2.drawContours(image,[y], -1, (0,255,255),3)
            lengthYellow +=1
            
    lengthGreen = 0
    for g in green_contours:
        area1 = cv2.contourArea(g)
        if area1>1600:
            # cv2.drawContours(image,[g], -1, (0,255,0),3)
            lengthGreen +=1

    lengthRed = 0
    for r in red_contours:
        area1 = cv2.contourArea(r)
        if area1>1600:
            # cv2.drawContours(image,[r], -1, (0,0,255),3)
            lengthRed +=1

    lengthBlue = 0
    for b in blue_contours:
        area1 = cv2.contourArea(b)
        if area1>1600:
            # cv2.drawContours(image,[b], -1, (255,0,0),3)
            lengthBlue +=1


    lengthOrange = 0
    for o in orange_contours:
        area1 = cv2.contourArea(o)
        if area1>1600:
            # cv2.drawContours(image,[o], -1, (24, 117, 255),3)
            lengthOrange +=1
    
    lengthBrown = 0
    for br in brown_contours:
        area1 = cv2.contourArea(br)
        if area1>1600:
            # cv2.drawContours(image,[br], -1, (19,69,139),3)
            lengthBrown +=1

    cv2.putText(image,f'Amarelo: {lengthYellow}', (20, 100),cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 255, 255),3 )
    cv2.putText(image,f'Verde: {lengthGreen}', (20, 160),cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 255, 0),3 )
    cv2.putText(image,f'Vermelho: {lengthRed}', (20, 220),cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 255),3 )
    cv2.putText(image,f'Azul: {lengthBlue}', (20, 280),cv2.FONT_HERSHEY_TRIPLEX, 2.0, (255, 0, 0),3 )
    cv2.putText(image,f'Laranja: {lengthOrange}', (20, 340),cv2.FONT_HERSHEY_TRIPLEX, 2.0, (24, 117, 255),3 )
    cv2.putText(image,f'Marrom: {lengthBrown}', (20, 400),cv2.FONT_HERSHEY_TRIPLEX, 2.0, (19,69,139),3 )
    return image

image = ContourCalculate(image)
cv2.imshow("Counting", image)
cv2.waitKey(0)