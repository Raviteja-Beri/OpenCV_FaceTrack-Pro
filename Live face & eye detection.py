import cv2

face_cascade = cv2.CascadeClassifier(r"C:\Users\HP\Downloads\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\HP\Downloads\haarcascade_eye.xml")

if face_cascade.empty():
    print("Error: Could not load face cascade classifier.")
    exit()

if eye_cascade.empty():
    print("Error: Could not load eye cascade classifier.")
    exit()

def detect_faces_and_eyes(gray, frame):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
   
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
       
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam opened successfully. Starting face and eye detection...")

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result_frame = detect_faces_and_eyes(gray, frame)

    cv2.imshow('Face and Eye Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

video_capture.release()
cv2.destroyAllWindows()
