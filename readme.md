Face and Signature Recognition App

This application captures and recognizes the user’s face using a webcam and the LBPH algorithm. After successful face recognition, a PyQt5 window opens where the user can draw their signature and verify it using a PyTorch-based trained model.

Features

Face capture and recognition with webcam
LBPH algorithm for face recognition
PyQt5 interface for signature drawing
Signature verification using a ResNet18 PyTorch model

Installation

Required libraries are opencv-python, opencv-contrib-python, PyQt5, torch, torchvision, pillow, and numpy.

How to Use

First, show your face to the camera; multiple face images will be saved automatically. Once your face is recognized, the signature drawing window opens. Draw your signature and click Check Signature to verify it. Use Retry to redraw your signature if needed.

Files and Folders

A folder stores face images and the face recognition model. A pre-trained signature verification model file is used during signature checking.