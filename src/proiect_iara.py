from datetime import datetime
import cv2
import glob
import os

from tkinter import *
from PIL import ImageTk, Image

class ARFilter():
    def __init__(self):
        self.path = '..\\resources\\'
        self.face_cascade = cv2.CascadeClassifier(self.path +'haar\\haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(self.path +'haar\\haarcascade_eye.xml')
        self.all_filters = glob.glob(self.path + "filters\\*")
        self.filter = self.all_filters[0]
        self.capture = cv2.VideoCapture(0)
        self.img = None
        self.init_form()
        print(f"\nSuccessfully initialized!\n")
        current_filter = str(self.filter).split("\\")[-1].split(".png")[0]
        print(f"\nCurrent filter: {current_filter}")

    # UI init + buttons
    def init_form(self):
        self.main_form = Tk(className= "Iara project")
        self.main_form.geometry("640x480")
        self.main_form.resizable(width=0, height=0)
        self.label = Label(self.main_form)
        self.label.grid(row = 0, column = 0)
        self.main_form.bind("<Key>", self.take_decision)
        left_button = Button(self.main_form, text = "<", command = self.change_left, width = 3, height = 1, font = "bold")
        left_button.place(x = 1, y = 480/2 - 20)
        right_button = Button(self.main_form, text = ">", command = self.change_right, width = 3, height = 1, font = "bold")
        right_button.place(x = 595, y = 480/2 - 20)
        snapshot_button = Button(self.main_form, text = "Take snapshot", command = self.take_snapshot, width = 15, height = 1, font = "bold")
        snapshot_button.place(x = 640/2 - 90, y = 440)

    # Treat binded events On Key Down 
    def take_decision(self, event):
        if(event.keycode == 27):
            exit(0)
        elif(event.keycode == 37):
            self.change_left()
        elif(event.keycode == 39):
            self.change_right()
        elif(event.keycode == 32):
            self.take_snapshot()
        
    # Take a snapshot to remember
    def take_snapshot(self):
        current_filter = str(self.filter).split("\\")[-1].split(".png")[0]
        filename = datetime.now().strftime(f"%Y_%m_%d-%H_%M_%S_{current_filter}.jpg")
        filepath = f"../resources/snapshots/{filename}"

        if not (os.path.exists("../resources/snapshots/")):
            os.makedirs("../resources/snapshots/")

        self.img.save(filepath)
        print(f"\nSuccesfully saved {filepath}")

    # Switch to the previous filter
    def change_left(self):
        length = len(self.all_filters)
        current_index = self.all_filters.index(self.filter)

        if(current_index) < 0:
            self.filter =  self.all_filters[length]
        elif(current_index > length):
            self.filter = self.all_filters[0]
        else:
            self.filter = self.all_filters[current_index-1]

        current_filter = str(self.filter).split("\\")[-1].split(".png")[0]
        print(f"\nCurrent filter: {current_filter}")

    # Switch to the next filter
    def change_right(self):
        length = len(self.all_filters)
        current_index = self.all_filters.index(self.filter)

        if(current_index) < 0:
            self.filter =  self.all_filters[length]
        elif(current_index >= length - 1):
            self.filter = self.all_filters[0]
        else:
            self.filter = self.all_filters[current_index+1]

        current_filter = str(self.filter).split("\\")[-1].split(".png")[0]
        print(f"\nCurrent filter: {current_filter}")


    # Capture frames and search for faces&eyes
    def show_frames(self, witch):
        b,g,r = cv2.split(witch)          
        witch = cv2.merge([r,g,b])
        # Get shape of filter
        original_witch_h, original_witch_w, _ = witch.shape

        # Convert to gray
        witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

        # Create mask and inverse mask of filter
        _, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        # Read each frame of video and convert to BGR
        img = self.capture.read()[1]
        b,g,r = cv2.split(img)          
        img = cv2.merge([r,g,b])
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find faces in image using classifier
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            # Coordinates of face region
            face_w = w
            face_h = h
            face_x1 = x
            face_x2 = face_x1 + face_w
            face_y1 = y
            face_y2 = face_y1 + face_h

            # Filter size in relation to face by scaling
            witch_width = int(1.5 * face_w)
            witch_height = int(witch_width * original_witch_h / original_witch_w)
            
            # Setting location of coordinates of filter
            witch_x1 = face_x2 - int(face_w / 2) - int(witch_width / 2)
            witch_x2 = witch_x1 + witch_width
            witch_y1 = face_y1 - int(face_h * 1.25)
            witch_y2 = witch_y1 + witch_height 

            # Check to see if out of frame
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

            # Resize filter to fit on face
            witch = cv2.resize(witch, (witch_width, witch_height), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(original_mask, (witch_width, witch_height), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(original_mask_inv, (witch_width, witch_height), interpolation = cv2.INTER_AREA)

            # Take ROI for witch from background that is equal to size of filter image
            roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

            # Original image in background (bg) where filter is not
            roi_bg = cv2.bitwise_and(roi, roi, mask = mask)
            roi_fg = cv2.bitwise_and(witch, witch, mask = mask_inv)
            dst = cv2.add(roi_bg, roi_fg)

            # Put back in original image
            img[witch_y1:witch_y2, witch_x1:witch_x2] = dst

        # Display image
        img = Image.fromarray(img)
        self.img = img
        imgtk = ImageTk.PhotoImage(image = img)
        self.label.imgtk = imgtk
        self.label.configure(image = imgtk)
        #  Repeat after an interval to captureture continiously
        self.label.after(20, self.show_frames, cv2.imread(self.filter))

# Get instances
if __name__ == "__main__":
    ar = ARFilter()
    ar.show_frames(cv2.imread(ar.filter))
    ar.main_form.mainloop()