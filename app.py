import cv2
import time
from deepface import DeepFace

def emotion_bravo(frame,x,y):
    logopng = cv2.imread('angry.png', cv2.IMREAD_UNCHANGED)
    logo_resized = cv2.resize(logopng, (100, 100))
    x, y = 150, 150
    for c in range(0, 3):
        frame[y:y + logo_resized.shape[0], x:x + logo_resized.shape[1], c] = \
            frame[y:y + logo_resized.shape[0], x:x + logo_resized.shape[1], c] * \
            (1 - logo_resized[:, :, 3] / 255.0) + \
            logo_resized[:, :, c] * (logo_resized[:, :, 3] / 255.0)

rostos_ = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('video.mp4')
# cap.set(cv2.CAP_PROP_FPS, 10)
xx =0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    time.sleep(0.02)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Display the resulting frame

    cv2.rectangle(frame, (10, 10), (100, 100), (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Usando IA em Imagens', (0+xx, 100), font, 1.0, (0, 255, 0), 2)
    xx+=5
    rosto = rostos_.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    print(rosto)
    for (x,y,w,h) in rosto:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        face_roi = rgb_frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        if emotion == 'angry':
            emotion_bravo(frame,x,y)
        else:
            cv2.putText(frame, emotion,(x,y),font,0.5,(0,255,0),2)




    
    
    
    
    cv2.imshow('Curso - ', frame)
   

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
