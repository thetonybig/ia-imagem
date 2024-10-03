import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
arma_cascade = cv2.CascadeClassifier('haarcascade_gun.xml')


def emotion_bravo(frame,x,y):
    logopng = cv2.imread('ungly.png', cv2.IMREAD_UNCHANGED)
    logo_resized = cv2.resize(logopng, (100, 100))
    x, y = 150, 150
    for c in range(0, 3):
        frame[y:y + logo_resized.shape[0], x:x + logo_resized.shape[1], c] = \
            frame[y:y + logo_resized.shape[0], x:x + logo_resized.shape[1], c] * \
            (1 - logo_resized[:, :, 3] / 255.0) + \
            logo_resized[:, :, c] * (logo_resized[:, :, 3] / 255.0)

# Start capturing video
#cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    persons = body_cascade.detectMultiScale(gray_frame)
    #armas = arma_cascade.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=2,minSize=(100,30))
    #for (x, y, w, h) in persons:
    #    gray = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #    font = cv2.FONT_HERSHEY_SIMPLEX
    #    cv2.putText(frame, 'person', (x, y - 10), font, 2.0, (11, 255, 255), 2)


    #for (x, y, w, h) in armas:
    #    gray = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #    font = cv2.FONT_HERSHEY_SIMPLEX
    #    cv2.putText(frame, 'arma', (x, y - 10), font, 2.0, (11, 255, 255), 2)

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    emotion_bravo(frame, 100, 100)
    # Display the resulting frame



    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
