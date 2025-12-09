from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from student import Student
from tkinter import messagebox
import mysql.connector
from time import strftime
from datetime import datetime
import cv2
import os
import numpy as np

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(self.root, text="FACE RECOGNITION", font=("times new roman", 35, "bold"), bg="white", fg="red")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # first image
        img_top = Image.open(r"College images/fp1.jpg")
        img_top = img_top.resize((650, 700), Image.LANCZOS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)
        f_lbl_left = Label(self.root, image=self.photoimg_top)
        f_lbl_left.place(x=0, y=55, width=650, height=700)

        # second image
        img_bottom = Image.open(r"College images/sp1.webp")
        img_bottom = img_bottom.resize((950, 700), Image.LANCZOS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)
        f_lbl_right = Label(self.root, image=self.photoimg_bottom)
        f_lbl_right.place(x=650, y=55, width=950, height=700)

        # button (connect to face_recog)
        b1_1 = Button(f_lbl_right, text="Face Recognition", cursor="hand2", command=self.face_recog,
                      font=("times new roman", 18, "bold"), bg="darkgreen", fg="white")
        b1_1.place(x=365, y=620, width=200, height=40)

    # attendance
    def mark_attendence(self, i, r, n, d):
        # Use a+ to create file if not exists; don't duplicate entries by checking id
        filename = "nandan.csv"
        try:
            with open(filename, "a+", newline="\n") as f:
                f.seek(0)
                myDataList = f.readlines()
                id_list = [line.split(",")[0].strip() for line in myDataList if line.strip()]
                # use i (student id) as unique key
                if str(i) not in id_list:
                    now = datetime.now()
                    d1 = now.strftime("%d/%m/%Y")
                    dtString = now.strftime("%H:%M:%S")
                    f.write(f"{i},{r},{n},{d},{dtString},{d1},Present\n")
        except Exception as e:
            print("Failed to write attendance:", e)

    # ========== face recognition ===========
    def face_recog(self):
        def draw_boundry(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

            coord = []

            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                try:
                    id_, predict = clf.predict(gray_image[y:y + h, x:x + w])
                    confidence = int((100 * (1 - predict / 300)))
                except Exception as e:
                    id_, confidence = None, 0

                # default values
                name_str = "Unknown"
                roll_str = "Unknown"
                dep_str = "Unknown"
                id_str = str(id_) if id_ is not None else "Unknown"

                # fetch details from DB safely
                try:
                    conn = mysql.connector.connect(host="localhost", username="root", password="Nandan@01",
                                                   database="face_recognizer")
                    my_cursor = conn.cursor()

                    if id_ is not None:
                        my_cursor.execute(
                            "SELECT Student_id, Name, Roll, Dep FROM student WHERE Student_id=%s",
                            (id_,)
                        )
                        row = my_cursor.fetchone()
                        if row:
                            id_str  = str(row[0])
                            name_str = str(row[1])
                            roll_str = str(row[2])
                            dep_str = str(row[3])


                        my_cursor.execute("SELECT Roll FROM student WHERE Student_id = %s", (id_,))
                        row = my_cursor.fetchone()
                        if row:
                            roll_str = str(row[0])

                        my_cursor.execute("SELECT Dep FROM student WHERE Student_id = %s", (id_,))
                        row = my_cursor.fetchone()
                        if row:
                            dep_str = str(row[0])

                        my_cursor.execute("SELECT Student_id FROM student WHERE Student_id = %s", (id_,))
                        row = my_cursor.fetchone()
                        if row:
                            id_str = str(row[0])

                    conn.close()
                except Exception as e:
                    print("DB error:", e)

                if confidence > 77:
                    cv2.putText(img, f"ID: {id_str}", (x, y - 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img, f"Roll: {roll_str}", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img, f"Name: {name_str}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img, f"Department: {dep_str}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    self.mark_attendence(id_str, roll_str,name_str,dep_str)
                    
                    # mark attendance
                    try:
                        self.mark_attendence(id_str, roll_str, name_str, dep_str)
                    except Exception as e:
                        print("Attendance error:", e)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(img, "Unknown Face", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

                coord = [x, y, w, h]

            return coord

        def recognize(img, clf, faceCascade):
            draw_boundry(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
            return img

        # load cascade
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            messagebox.showerror("Error", f"Haar cascade not found: {cascade_path}")
            return

        faceCascade = cv2.CascadeClassifier(cascade_path)

        # create LBPH recognizer (requires opencv-contrib-python)
        try:
            clf = cv2.face.LBPHFaceRecognizer_create()
        except Exception as e:
            messagebox.showerror("Error", "cv2.face.LBPHFaceRecognizer_create() not available. "
                                          "Install opencv-contrib-python.")
            print("OpenCV face module error:", e)
            return

        # load trained classifier
        classifier_file = "classifier.xml"
        if not os.path.exists(classifier_file):
            messagebox.showerror("Error", f"Classifier file not found: {classifier_file}")
            return

        clf.read(classifier_file)

        # start video capture
        video_cap = cv2.VideoCapture(0)
        if not video_cap.isOpened():
            messagebox.showerror("Error", "Unable to open webcam.")
            return

        while True:
            ret, img = video_cap.read()
            if not ret:
                print("Failed to read from webcam.")
                break

            img = recognize(img, clf, faceCascade)
            cv2.imshow("Welcome to Face Recognition", img)

            # press Enter (keycode 13) to exit
            if cv2.waitKey(1) == 13:
                break

        video_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()
