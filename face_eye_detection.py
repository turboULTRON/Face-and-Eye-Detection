import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class FaceEyeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Eye Detection")
        self.root.geometry("800x600")

        self.image_label = Label(self.root)
        self.image_label.pack()

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        
        self.webcam_button = Button(root, text="Use Webcam", command=self.use_webcam)
        self.webcam_button.pack()
        
        self.quit_button = Button(root, text="Quit", command=root.quit)
        self.quit_button.pack()
        
    def upload_image(self):
        file_path= filedialog.askopenfilename(filetypes=[("Image File",'*.jpg;*.png;*.jpeg')])
        if file_path:
            image= cv2.imread
            self.detect_faces_and_eyes(image, display=True)
            
    def use_webcam(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.detect_faces_and_eyes(frame, display=False)
            cv2.imshow("Webcam - Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def detect_faces_and_eyes(self, frame, display):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
        if display:
            #convert the image to RGB format
            rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #convert the image to PIL format
            pil_image= Image.fromarray(rgb_frame)
            #convert the image to ImageTk format
            tk_image= ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=tk_image)
            self.image_label.image= tk_image
            
# Initialize the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEyeDetectionApp(root)
    root.mainloop()