import cv2
import os
import numpy as np
import time
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QBuffer
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define folder to save face images
image_folder = os.path.join(os.path.expanduser("~"), "faces_data")
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

user_name = "Hacer"
user_folder = os.path.join(image_folder, user_name)

if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# Capture face images if less than 15 images saved
if len(os.listdir(user_folder)) < 15:
    cap = cv2.VideoCapture(0)

    print(f"Saving {user_name}'s face images. Please look at the camera and slightly move your face.")
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            count += 1
            # Save face image
            cv2.imwrite(os.path.join(user_folder, f"user_{count}.jpg"), face_img)
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Saving Faces", frame)

        if count >= 15:
            print("15 face images saved. Done.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("Faces are already saved.")

# Prepare data for training the recognizer
def prepare_data():
    faces = []
    labels = []
    label_map = {}
    label_id = 0
    for user in os.listdir(image_folder):
        user_folder_path = os.path.join(image_folder, user)
        if os.path.isdir(user_folder_path):
            for image_name in os.listdir(user_folder_path):
                image_path = os.path.join(user_folder_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces.append(image)
                if user not in label_map:
                    label_map[user] = label_id
                    label_id += 1
                labels.append(label_map[user])
    return faces, labels, label_map

faces, labels, label_map = prepare_data()

# Train the recognizer with faces and labels
recognizer.train(faces, np.array(labels))
recognizer.save(os.path.join(image_folder, "face_recognizer.yml"))

# Start video capture for face recognition
cap = cv2.VideoCapture(0)

print(f"Starting face recognition. Only {user_name}'s face will be recognized.")
start_time = None
face_recognized = False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face_img)

        # If recognized face is the user and confidence is good
        if label == label_map[user_name] and confidence < 50:
            cv2.putText(frame, f"{user_name} Recognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if start_time is None:
                start_time = time.time()

            # If face recognized continuously for 5 seconds, proceed
            if time.time() - start_time >= 5:
                face_recognized = True
                print("Face recognized, starting signature verification.")
                cap.release()
                cv2.destroyAllWindows()
                break

        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or face_recognized:
        break

cap.release()
cv2.destroyAllWindows()

if face_recognized:
    # Define the signature recognition model (ResNet18 based)
    class SignatureModel(nn.Module):
        def __init__(self):
            super(SignatureModel, self).__init__()
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 1)

        def forward(self, x):
            return self.model(x)

    # Load the trained signature model
    model_path = os.path.join(os.path.expanduser("~"), "imza", "imza_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # Image transformations for signature input
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # PyQt window for signature drawing and verification
    class DrawingWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Signature Drawing")
            self.setGeometry(100, 100, 400, 400)
            self.image = QPixmap(self.size())
            self.image.fill(Qt.white)
            self.drawing = False
            self.last_point = QPoint()

            layout = QVBoxLayout()
            self.check_button = QPushButton("Check Signature", self)
            self.check_button.clicked.connect(self.check_signature)
            layout.addWidget(self.check_button)

            self.retry_button = QPushButton("Retry", self)
            self.retry_button.clicked.connect(self.retry)
            layout.addWidget(self.retry_button)

            self.result_label = QLabel(self)
            layout.addWidget(self.result_label)

            self.setLayout(layout)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self.image, self.image.rect())

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.last_point = event.pos()

        def mouseMoveEvent(self, event):
            if event.buttons() & Qt.LeftButton and self.drawing:
                painter = QPainter(self.image)
                painter.setPen(QPen(Qt.black, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.last_point, event.pos())
                self.last_point = event.pos()
                self.update()

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.drawing = False

        def check_signature(self):
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)
            self.image.save(buffer, "PNG")
            pil_image = Image.open(io.BytesIO(buffer.data()))
            tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prediction = torch.sigmoid(output).item()

            if prediction > 0.5:
                self.result_label.setText("RECOGNIZED (Genuine)")
            else:
                self.result_label.setText("NOT RECOGNIZED (Fake)")

        def retry(self):
            self.image.fill(Qt.white)
            self.update()
            self.result_label.clear()

    # Main window to show the drawing window
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Signature Verification App")
            self.setGeometry(100, 100, 450, 500)
            self.drawing_window = DrawingWindow()
            self.setCentralWidget(self.drawing_window)

    if __name__ == "__main__":
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())
