from typing import Match
import cv2
import glob

from tkinter import *
from PIL import ImageTk, Image


def take_decision(event):
    if(event.keycode == 27):
        exit(0)
    elif(event.keycode == 37):
        change_left()
    elif(event.keycode == 39):
        change_right()



win = Tk(className= "Iara")
win.geometry("640x480")
win.resizable(width=0, height=0)
label = Label(win)
label.grid(row = 0, column = 0)
win.bind("<Key>", take_decision)

# Facial/eyes classifiers
path = '..\\resources\\'
face_cascade = cv2.CascadeClassifier(path +'haar\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haar\\haarcascade_eye.xml')

all_filters = glob.glob(path + "filters\\*")
witch = all_filters[1]
filter = all_filters[1]

# Switch to last filter
def change_left():
    global witch
    global all_filters
    length = len(all_filters)
    current_index = all_filters.index(witch)

    if(current_index) < 0:
        witch =  all_filters[length]
    elif(current_index > length):
        witch = all_filters[0]
    else:
        witch = all_filters[current_index-1]
    print(witch)


# Switch to next filter
def change_right():
    global witch
    global all_filters
    length = len(all_filters)
    current_index = all_filters.index(witch)

    if(current_index) < 0:
        witch =  all_filters[length]
    elif(current_index >= length - 1):
        witch = all_filters[0]
    else:
        witch = all_filters[current_index+1]
    print(witch)


left_button = Button(win, text = "<", command = change_left)
left_button.place(x = 0, y = 480/2)

right_button = Button(win, text = ">", command=change_right)
right_button.place(x = 620, y = 480/2)






# Get shape of witch
original_witch_h, original_witch_w, witch_channels = cv2.imread(witch).shape

# Convert to gray
witch_gray = cv2.cvtColor(cv2.imread(witch), cv2.COLOR_BGR2GRAY)

# create mask and inverse mask of witch
ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

# Read video
cap = cv2.VideoCapture(0)
_, img = cap.read()
img_h, img_w = img.shape[:2]


# Capture frames and search for faces&eyes
def show_frames(witch):
    # Read each frame of video and convert to gray
    img = cap.read()[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find faces in image using classifier
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # for every face found:
    for (x, y, w, h) in faces:
        # coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        # witch size in relation to face by scaling
        witch_width = int(1.5 * face_w)
        witch_height = int(witch_width * original_witch_h / original_witch_w)
        
        # setting location of coordinates of witch
        witch_x1 = face_x2 - int(face_w / 2) - int(witch_width / 2)
        witch_x2 = witch_x1 + witch_width
        witch_y1 = face_y1 - int(face_h * 1.25)
        witch_y2 = witch_y1 + witch_height 

        # check to see if out of frame
        if witch_x1 < 0:
            witch_x1 = 0
        if witch_y1 < 0:
            witch_y1 = 0
        if witch_x2 > img_w:
            witch_x2 = img_w
        if witch_y2 > img_h:
            witch_y2 = img_h

        # Account for any out of frame changes
        witch_width = witch_x2 - witch_x1
        witch_height = witch_y2 - witch_y1

        # resize witch to fit on face
        witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

        # take ROI for witch from background that is equal to size of witch image
        roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

        # original image in background (bg) where witch is not
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(witch,witch,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        # put back in original image
        img[witch_y1:witch_y2, witch_x1:witch_x2] = dst
        break

    # display image
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure(image = imgtk)
    #  Repeat after an interval to capture continiously
    label.after(20, show_frames, cv2.imread(witch))
    
show_frames(cv2.imread(witch))
win.mainloop()
