from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Hardcoded username and password (Replace this with a proper authentication mechanism)
valid_credentials = {'admin': 'password'}

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.json.get('username')
    password = request.json.get('password')

    if username in valid_credentials and valid_credentials[username] == password:
        return jsonify({'authenticated': True})
    else:
        return jsonify({'authenticated': False})

@app.route('/dataset_maker')
def dataset_maker():
    # Here, you can add further authorization checks if needed
    return render_template('dataset_maker.html')

if __name__ == '__main__':
    app.run(debug=True)
