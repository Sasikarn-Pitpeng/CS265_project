# ref : https://github.com/amartya-k/vision
# Adapted : SoalakAI (AI in Government)
# CS265 project
import cv2
import numpy as np
import vehicles # from vehicles.py
import time

cnt_up=0
cnt_down=0
cnt_all = 0

#cap=cv2.VideoCapture("Freewa.mp4")
#cap=cv2.VideoCapture("surveillance.m4v")
# cap=cv2.VideoCapture("diy_test.mp4")
cap=cv2.VideoCapture("video.mp4")

#Get width and height of video
w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=frameArea/400

#Lines
line_up=int(2*(h/3))
line_down=int(2*(h/3))

up_limit=int(line_up - 150)
down_limit=int(line_up + 150)

print("Red line y:",str(line_down))
print("Blue line y:",str(line_up))

line_down_color=(0,0,255) # set line colour to red (b,g,r)
line_up_color=(255,0,0) # set line colour to blue (b,g,r)

pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#Kernals
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)


font = cv2.FONT_HERSHEY_DUPLEX 
cars = []
max_p_age = 5
pid = 1


while(cap.isOpened()):
    ret,frame=cap.read()
    for i in cars:
        i.age_one()
    fgmask=fgbg.apply(frame)
    fgmask2=fgbg.apply(frame)

    if ret==True:

        #Binarization
        ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2=cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #OPening i.e First Erode the dilate
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
        mask2=cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)

        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)


        #Find Contours
        countours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area=cv2.contourArea(cnt)
            #print(area)
            if area>areaTH:
                ####Tracking######
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt)

                new=True
                if cy in range(up_limit,down_limit):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h: # if car is in the video
                            new = False
                            i.updateCoords(cx, cy) # then update the coordinate

                            if i.going_UP(line_down,line_up)==True: # if it go up
                                cnt_up+=1 
                                cnt_all+=1
                                cv2.polylines(frame, [pts_L2], False, line_down_color, thickness=3) # change the colour of the checking line -> make it flashing
                                #print("ID:",i.getId(),'crossed going up at', time.strftime("%c"))
                            elif i.going_DOWN(line_down,line_up)==True: # same as go up
                                cnt_down+=1
                                cnt_all+=1
                                cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=3)
                                #print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                            break
                        if i.getState()=='1': # if the car was passed the checking line
                            if i.getDir()=='down'and i.getY()>down_limit: # and the car was passed the down limit line
                                i.setDone() # then set done -> do not count anymore
                            elif i.getDir()=='up'and i.getY()<up_limit: # same as above
                                i.setDone()
                        if i.timedOut():
                            index=cars.index(i)
                            cars.pop(index)
                            del i

                    if new==True: #If nothing is detected,create new
                        p=vehicles.Car(pid,cx,cy,max_p_age)
                        cars.append(p)
                        pid+=1

                cv2.circle(frame,(cx,cy),5,(0,0,255),-1) # draw a red circle in the centre
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # draw a green regtangle around the detected car

        for i in cars:
            cv2.putText(frame, str(i.getRGB()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)

        str_up='UP: '+str(cnt_up) #create a string to show how many cars going up
        str_down='DOWN: '+str(cnt_down) #create a string to show how many cars going up
        str_all = 'ALL:'+str(cnt_all) #create a string to show how many were detected

        frame=cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2) # draw the going-up checking line
        frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2) # draw the going-down checking line
        frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1) # draw going-up limit line
        frame=cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1) # draw going-down limit line

        cv2.putText(frame, str_up, (10, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, str_up, (10, 40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (160, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, str_down, (160, 40), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str_all, (360, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, str_all, (360, 40), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Press "Q" to exit.', (10, 80), font, 1, (0, 0, 0), 2, cv2.LINE_AA) #contour of the text
        cv2.putText(frame, 'Press "Q" to exit.', (10, 80), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Frame',frame)

        if cv2.waitKey(1) & 0xff==ord('q'):
            break

    else:
        break
# print conclusion in terminal when exit.
print()
print('CONCLUSION')
print(' UP:',int(cnt_up))
print(' DOWN:',int(cnt_down))
print(' ALL:',int(cnt_all))

cap.release()
cv2.destroyAllWindows()