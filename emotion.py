import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
#arma_cascade = cv2.CascadeClassifier('haarcascade_gun.xml')

#print(cv2.data.haarcascades)
# Start capturing video
cap = cv2.VideoCapture('video.mp4')
#cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    persons = body_cascade.detectMultiScale(gray_frame)
    #armas = arma_cascade.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=2,minSize=(30,30))
    #for (x, y, w, h) in persons:
    #    gray = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #    font = cv2.FONT_HERSHEY_SIMPLEX
    #    cv2.putText(frame, 'person', (x, y - 10), font, 2.0, (11, 255, 255), 2)


   # for (x, y, w, h) in armas:
   #     gray = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
   #     font = cv2.FONT_HERSHEY_SIMPLEX
   #     cv2.putText(frame, 'arma', (x, y - 10), font, 2.0, (11, 255, 255), 2)

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        #result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        #emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
