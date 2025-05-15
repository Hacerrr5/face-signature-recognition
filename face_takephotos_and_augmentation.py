import cv2
import os
import numpy as np
import time

# Folder to save face images
output_folder = "photos"
os.makedirs(output_folder, exist_ok=True)

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start camera
camera = cv2.VideoCapture(0)

count = 0
max_original_images = 100  # Number of original images to capture
max_total_images = 1000    # Total images including augmented ones

last_time = 0

def augment(image):
    """
    Create augmented versions of the given image.
    Returns a list of augmented images.
    """
    augmented = []
    flipped = cv2.flip(image, 1)
    brighter = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    darker = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
    noise = np.random.randint(0, 30, image.shape, dtype='uint8')
    noisy = cv2.add(image, noise)

    augmented.extend([flipped, brighter, darker, noisy])
    return augmented

print("Get ready! Face capture starting...")

while count < max_original_images:
    ret, frame = camera.read()
    if not ret:
        continue

    current_time = time.time()
    if current_time - last_time >= 5:  # Capture every 5 seconds
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            # Save original face image
            original_filename = f"{count}_original.jpg"
            cv2.imwrite(os.path.join(output_folder, original_filename), face)

            # Generate and save augmented images for each original
            augmented_images = augment(face)
            aug_count = 0
            for img in augmented_images:
                if count * len(augmented_images) + aug_count >= max_total_images - max_original_images:
                    break
                aug_filename = f"{count}_aug_{aug_count}.jpg"
                cv2.imwrite(os.path.join(output_folder, aug_filename), img)
                aug_count += 1

            count += 1
            print(f"Saved original image #{count} and {aug_count} augmented images.")
            last_time = current_time

            break  # Process only one face per frame

    cv2.imshow("Face Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Done! Face images are ready.")
