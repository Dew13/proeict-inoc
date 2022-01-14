import cv2
import glob

from tkinter import *
from PIL import ImageTk, Image

class ARFilter():
    def __init__(self):
        # Facial/eyes classifiers
        self.path = '..\\resources\\'
        self.face_cascade = cv2.CascadeClassifier(self.path +'haar\\haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(self.path +'haar\\haarcascade_eye.xml')
        self.all_filters = glob.glob(self.path + "filters\\*")
        self.filter = self.all_filters[1]
        self.capture = cv2.VideoCapture(0)
        self.init_form()
        print("Successfully initialized!")


    def init_form(self):
        self.main_form = Tk(className= "Iara")
        self.main_form.geometry("640x480")
        self.main_form.resizable(width=0, height=0)
        self.label = Label(self.main_form)
        self.label.grid(row = 0, column = 0)
        self.main_form.bind("<Key>", self.take_decision)
        left_button = Button(self.main_form, text = "<", command = self.change_left)
        left_button.place(x = 0, y = 480/2)
        right_button = Button(self.main_form, text = ">", command = self.change_right)
        right_button.place(x = 620, y = 480/2)


    def take_decision(self, event):
        if(event.keycode == 27):
            exit(0)
        elif(event.keycode == 37):
            self.change_left()
        elif(event.keycode == 39):
            self.change_right()


    def change_left(self):
        length = len(self.all_filters)
        current_index = self.all_filters.index(self.filter)

        if(current_index) < 0:
            self.filter =  self.all_filters[length]
        elif(current_index > length):
            self.filter = self.all_filters[0]
        else:
            self.filter = self.all_filters[current_index-1]
        print(self.filter)


    def change_right(self):
        length = len(self.all_filters)
        current_index = self.all_filters.index(self.filter)

        if(current_index) < 0:
            self.filter =  self.all_filters[length]
        elif(current_index >= length - 1):
            self.filter = self.all_filters[0]
        else:
            self.filter = self.all_filters[current_index+1]
        print(self.filter)


    # Capture frames and search for faces&eyes
    def show_frames(self, witch):
        # Get shape of witch
        original_witch_h, original_witch_w, _ = witch.shape

        # Convert to gray
        witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

        # Create mask and inverse mask of witch
        _, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)
        # Read each frame of video and convert to gray
        img = self.capture.read()[1]
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

            # Resize witch to fit on face
            witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

            # Take ROI for witch from background that is equal to size of witch image
            roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

            # Original image in background (bg) where witch is not
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
            roi_fg = cv2.bitwise_and(witch, witch,mask=mask_inv)
            dst = cv2.add(roi_bg,roi_fg)

            # Put back in original image
            img[witch_y1:witch_y2, witch_x1:witch_x2] = dst
            break

        # Display image
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image = img)
        self.label.imgtk = imgtk
        self.label.configure(image = imgtk)
        #  Repeat after an interval to captureture continiously
        self.label.after(20, self.show_frames, cv2.imread(self.filter))

ar = ARFilter()
ar.show_frames(cv2.imread(ar.filter))
ar.main_form.mainloop()