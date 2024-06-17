import cv2
import numpy as np
import json

def load_dataset_info():
    with open('dataset_info.json', 'r') as f:
        return json.load(f)

def load_images_and_labels_from_info(dataset_info):
    images = []
    labels = []
    label_dict = {}
    label_id = 0

    for label, data in dataset_info.items():
        for item in data:
            img = cv2.imread(item['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_id)
        label_dict[label_id] = label
        label_id += 1

    return images, labels, label_dict, dataset_info

def display_information(im, label_id, label_dict, dataset_info, x, y, w, h):
    label = label_dict[label_id]
    info = dataset_info[label]
    gender = info[0]['gender']
    program = info[0]['program']

    # Calculate text position for right side of the detected face
    text_position_x = min(im.shape[1] - 10, x + w + 10)  # Adjust the position as needed
    text_position_y = y + 30

    # Display name, gender, and program
    cv2.putText(im, f'Name: {label}', (x - 10, y - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (51, 51, 255), 2)
    cv2.putText(im, f'Gender: {gender}', (text_position_x, text_position_y), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (51, 51, 255), 2)
    cv2.putText(im, f'Program: {program}', (text_position_x, text_position_y + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (51, 51, 255), 2)

if __name__ == "__main__":
    # Load dataset information
    dataset_info = load_dataset_info()

    # Load training images, labels, label dictionary, and dataset info
    images, labels, label_dict, dataset_info = load_images_and_labels_from_info(dataset_info)

    # Train the face recognizer model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.array(labels))

    # Initialize video capture object
    webcam = cv2.VideoCapture(0)

    # Load the pre-trained Haar Cascade classifier for face detection from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame from the webcam
        ret, im = webcam.read()
        if not ret:
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Haar Cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))

            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1] < 70:  # Adjust recognition threshold as needed
                display_information(im, prediction[0], label_dict, dataset_info, x, y, w, h)
            else:
                cv2.putText(im, 'NOT RECOGNIZED', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('Recognizer', im)

        key = cv2.waitKey(10)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()
