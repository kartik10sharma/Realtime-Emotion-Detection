import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize face cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Variables to store emotion data
emotion_data = defaultdict(list)
time_data = []
frame_count = 0
skip_frames = 5
last_emotions = {}

# Define a scoring system
emotion_scores = {
    'happy': 3,
    'neutral': 1,
    'surprise': 2,
    'sad': -2,
    'angry': -3,
    'disgust': -3,
    'fear': -1,
}

# Variable to store total score
total_score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam. Exiting...")
        break

    # Resize frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Analyze emotions every N frames
    if frame_count % skip_frames == 0:
        try:
            result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)
            if result:
                last_emotions = result[0]['emotion']
                for emotion, value in last_emotions.items():
                    emotion_data[emotion].append(value)
                time_data.append(frame_count)
                # Update total score based on detected emotions
                dominant_emotion = max(last_emotions, key=last_emotions.get, default=None)
                if dominant_emotion and dominant_emotion in emotion_scores:
                    total_score += emotion_scores[dominant_emotion]
            else:
                for emotion in emotion_data.keys():
                    emotion_data[emotion].append(0)
                time_data.append(frame_count)
        except Exception as e:
            print("Error: " + str(e))
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        x, y, w, h = [int(v * 2) for v in (x, y, w, h)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display dominant emotion
    dominant_emotion = max(last_emotions, key=last_emotions.get, default="No face detected")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, dominant_emotion, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
    
    # Show frame
    cv2.imshow('Original video', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()

# Plot emotion data vs. time
plt.figure(figsize=(10, 6))
for emotion, values in emotion_data.items():
    plt.plot(time_data, values, label=emotion)

plt.title("Emotion Analysis Over Time")
plt.xlabel("Frame Count")
plt.ylabel("Emotion Probability")
plt.legend()
plt.show()

# Print the total score
print(f"Total Score for the candidate: {total_score}")
