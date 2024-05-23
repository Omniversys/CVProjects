import cv2
import numpy as np

# Load the cascade classifier detection object
cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
# Turn on the web camera
video_capture = cv2.VideoCapture(0)

while True:
    # Read data from the web camera (get the frame)
    _, frame = video_capture.read()
# Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Predict the bounding box of the eyes
    boxes = cascade.detectMultiScale(gray, 1.3, 10)
# Filter out images taken from a bad angle with errors
# We want to make sure both eyes were detected, and nothing else
    if len(boxes) == 2:
        eyes = []
        for box in boxes:
    # Get the rectangle parameters for the detected eye
            x, y, w, h = box
        # Crop the bounding box from the frame
            eye = frame[y:y + h, x:x + w]
        # Resize the crop to 32x32
            eye = cv2.resize(eye, (32, 32))
        # Normalize
            eye = (eye - eye.min()) / (eye.max() - eye.min())
    # Further crop to just around the eyeball
            eye = eye[10:-10, 5:-5]
    # Scale between [0, 255] and convert to int datatype
            eye = (eye * 255).astype(np.uint8)
    # Add the current eye to the list of 2 eyes
            eyes.append(eye)
  # Concatenate the two eye images into one
        eyes = np.hstack(eyes)
        filename= "test.jpg"
        cv2.imwrite(filename, eyes)

    cv2.imshow("Demo", frame)
    
    if cv2.waitKey(1) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()