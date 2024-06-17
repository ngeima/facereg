from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)

# In-memory user credentials storage
user_credentials = {'admin': 'password'}

# Dataset information storage
dataset_info = {}

def load_dataset_info():
    if os.path.exists('dataset_info.json'):
        with open('dataset_info.json', 'r') as f:
            return json.load(f)
    return {}

def save_dataset_info(dataset_info):
    with open('dataset_info.json', 'w') as f:
        json.dump(dataset_info, f)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in user_credentials:
            return jsonify({'registered': False, 'message': 'Username already exists'})
        
        user_credentials[username] = password
        return jsonify({'registered': True})
    return render_template('register.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.json.get('username')
    password = request.json.get('password')

    if username in user_credentials and user_credentials[username] == password:
        return jsonify({'authenticated': True})
    else:
        return jsonify({'authenticated': False})

@app.route('/dataset_maker')
def dataset_maker():
    return render_template('dataset_maker.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.json
    name = data['name']
    gender = data['gender']
    program = data['program']
    image_data = data['image']

    # Decode the image data
    image_data = image_data.split(',')[1]
    image_data = base64.b64decode(image_data)
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Save the image to disk
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    image_path = os.path.join('uploads', f"{name}.png")
    cv2.imwrite(image_path, image)

    # Update dataset info
    dataset_info = load_dataset_info()
    if name not in dataset_info:
        dataset_info[name] = []
    dataset_info[name].append({'image_path': image_path, 'gender': gender, 'program': program})
    save_dataset_info(dataset_info)

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
