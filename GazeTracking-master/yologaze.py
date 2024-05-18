import cv2
import numpy as np
import ultralytics

# Load the YOLOv8 model
model = ultralytics.YOLO("yolo-Weights/yolov8n.pt")


# Load the video stream
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 768)

# Define the gaze tracking function
def gaze_tracking(frame):
    # Detect objects in the frame
    detections = model(frame, stream=True)

    # Get the bounding boxes of the detected objects
    for r in detections:
        boxes = r.boxes
        
        for box in boxes:
            cls =int(box.cls[0])
            print (cls)
            if box.cls == 'eye':
                
                # Calculate the center of the eye bounding boxes
                eye_centers = [(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2]

            # Draw the gaze tracking results on the frame
                for eye_center in eye_centers:
                    cv2.circle(frame, eye_center, 5, (0, 255, 0), 2)
        # Filter out the bounding boxes that are not of the 'eye' class
        # eye_boxes = [box for box in boxes if box.cls == 'eye']


    # Return the frame with the gaze tracking results
    return frame

# Start the gaze tracking loop
while True:

    # Read a frame from the video stream
    ret, frame = cap.read()

    # If the frame is empty, break the loop
    if not ret:
        break

    # Perform gaze tracking on the frame
    frame = gaze_tracking(frame)

    # Display the frame with the gaze tracking results
    cv2.imshow('Gaze Tracking', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
cap.release()

# Close all windows
cv2.destroyAllWindows()