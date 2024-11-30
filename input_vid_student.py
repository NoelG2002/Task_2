import numpy as np
import cv2
from datetime import datetime, time
import os

from tensorflow.keras.models import model_from_json

# student_recognition_model
with open("face_recognition_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("face_recognition_weights.weights.h5")

# emotion_model
with open("model_a1.json", "r") as json_file:
    loaded_emotion_model_json = json_file.read()

emotion_model = model_from_json(loaded_emotion_model_json)
emotion_model.load_weights("model_weights1.weights.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 64

def recognize_and_detect_emotions(frame, label_dict, marked_students):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_img = gray_frame[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img_resized = np.expand_dims(face_img_resized, axis=-1)
        face_img_resized = np.expand_dims(face_img_resized, axis=0) / 255.0  # Normalize

        student_prediction = loaded_model.predict(face_img_resized)
        recognized_student_id = np.argmax(student_prediction)
        student_name = label_dict[recognized_student_id]

        emotion_prediction = emotion_model.predict(face_img_resized)
        recognized_emotion_id = np.argmax(emotion_prediction)
        emotions_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
        recognized_emotion = emotions_dict[recognized_emotion_id]

        cv2.putText(frame, f"Student: {student_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, f"Emotion: {recognized_emotion}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Log the attendance if the student hasn't been marked yet
        if student_name not in marked_students:
            log_attendance(student_name, recognized_emotion)
            marked_students.add(student_name)  # Mark the student as present

    return frame

# Log attendance to a CSV file
def log_attendance(student_name, emotion):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance_data = f"{current_time},{student_name},{emotion}\n"
    
    with open("attendance_log.csv", "a") as f:
        f.write(attendance_data)

cap = cv2.VideoCapture(0)  

# Attendance time window
start_time = time(9, 30)
end_time = time(10, 0)

label_dict = {0: 'Alejandro_Toledo', 1: 'Alvaro_Uribe', 2: 'Andre_Agassi', 3: 'Ariel_Sharon',4: 'Gloria_Macpagal_Arroyo',
              5: 'Guillermo_Coria', 6: 'Hans_Blix',7: 'Hugo_Chavez',8: 'Jacques_Chirac',9: 'Jean_Chreitien',10: 'Jennifer_Capriati',
              11:'John_Ashcroft',12: 'John_Negroponte',13: 'Jose_Maria_Aznar',14: 'Juan_Carlos_Ferrero',15: 'Junichiro_Koizumi',
              16: 'Kofi_Annan',17: 'Laura_Bush',18: 'Lindsay_Davenport',19: 'Lleyton_Hewitt',20: 'Luiz_Inacio_da_Silva',
              21: 'Megawati_Sukarnoputri',22: 'Nestor_Kirchnar',23: 'Pete_Sampras',24: 'Ricardo_Lagos',25: 'Roh_Moo-hyun',
              26: 'Silvio_Berlusconi',27:' Tom_Daschle',28: 'Tom_Ridge',29: 'Vicente_Fox'}  

marked_students = set()

while True:
    current_time = datetime.now().time()

    if start_time <= current_time <= end_time:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame.")
            break

        processed_frame = recognize_and_detect_emotions(frame, label_dict, marked_students)
        
        cv2.imshow("Attendance and Emotion Detection", processed_frame)

    else:
        print("Outside of attendance window.")
    
    # Press 'e' to quit the video capture
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
