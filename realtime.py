import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#  video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    
    # Analyze the frame for emotions
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result:  # Check if any face is detected
            dominant_emotion = result[0]['dominant_emotion']
        else:
            dominant_emotion = "No face detected"
    except Exception as e:
        dominant_emotion = "Error: " + str(e)
    
    # frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    #  rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # dominant emotion on  frames
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
    
    #  rectangles and emotion
    cv2.imshow('Original video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
