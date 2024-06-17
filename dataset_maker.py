import cv2
import os
import json

def capture_images_and_labels():
    # Prompt user to enter the name, gender, and program of the person
    person_name = input("Enter the Name of the person: ")
    person_gender = input("Enter the Gender of the person: ")
    person_program = input("Enter the Program of the person: ")

    # Create a subdirectory within the dataset folder named after the person
    dataset_folder = 'dataset'
    person_folder = os.path.join(dataset_folder, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    # Load the pre-trained Haar Cascade classifier for face detection from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the video capture object (0 for the default webcam)
    cap = cv2.VideoCapture(0)


    image_counter = 0
    max_images = 30

    # Initialize dictionary to store labels and image paths along with additional information
    data = {}

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the frame to grayscale (Haar Cascade works with grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces and save them
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            image_path = os.path.join(person_folder, f"{person_name}_{image_counter}.jpg")
            cv2.imwrite(image_path, face)
            image_counter += 1

            # Add label, image path, gender, and program to data dictionary
            data.setdefault(person_name, []).append({
                'image_path': image_path,
                'gender': person_gender,
                'program': person_program
            })

            # Display the name of the person if face is detected
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Break the loop if reached the maximum number of images
            if image_counter >= max_images:
                break

        # Display the resulting frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Break the loop if reached the maximum number of images
        if image_counter >= max_images:
            break

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Serialize and save the data to a JSON file
    with open('dataset_info.json', 'w') as f:
        json.dump(data, f)

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images_and_labels()
