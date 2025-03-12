import bcrypt
from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file, session
import mysql.connector, os
#Import libraries
import keras
import cv2
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import bcrypt


# Define constants
IMG_SIZE = 224  # Image size for InceptionV3
MAX_SEQ_LENGTH = 20  # Adjusted to match model's expected sequence length
NUM_FEATURES = 2048  # Features from InceptionV3 (features per frame)


# Load the model
model = load_model('GRU.h5')

# Function to load and preprocess a single video
def load_single_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # Convert from BGR to RGB
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Crop the center square of the frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Preprocess the frames (extract features using InceptionV3)
def preprocess_frames(frames):
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    frames = preprocess_input(frames)  # Preprocess the frames
    features = feature_extractor.predict(frames)
    return features



app = Flask(__name__)
app.secret_key =  "Your key"

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='video_dataset1'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/About')
def about():
    return render_template('about.html')

# @app.route('/register', methods=["GET", "POST"])
# def register():
#     if request.method == "POST":
#         email = request.form['useremail']
#         password = request.form['password']
#         c_password = request.form['c_password']
#         username = request.form['username']
#         age = request.form['age']
#         gender = request.form['gender']
#         if password == c_password:
#             query = "SELECT UPPER(email) FROM users"
#             email_data = retrivequery2(query)
#             email_data_list = []
#             for i in email_data:
#                 email_data_list.append(i[0])
#             if email.upper() not in email_data_list:
#                 # hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#                 query = "INSERT INTO users (email, password, username, age, gender) VALUES (%s, %s, %s, %s, %s)"
#                 values = (email, password, username, age, gender)
#                 executionquery(query, values)
#                 return render_template('login.html', message="Successfully Registered! Please go to login section")
#             return render_template('register.html', message="This email ID is already exists!")
#         return render_template('register.html', message="Conform password is not match!")
#     return render_template('register.html')

# import bcrypt
import requests

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['useremail']
        password = request.form['password']
        c_password = request.form['c_password']
        username = request.form['username']
        age = request.form['age']
        gender = request.form['gender']

        if password == c_password:
            query = "SELECT UPPER(email) FROM users1"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]

            if email.upper() not in email_data_list:
               
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                query = "INSERT INTO users1 (email, password, username, age, gender) VALUES (%s, %s, %s, %s, %s)"
                values = (email, hashed_password, username, age, gender)
                executionquery(query, values)

                return render_template('login.html', message="Successfully Registered! Please go to login section")
            
            return render_template('register.html', message="This email ID already exists!")

        return render_template('register.html', message="Confirm password does not match!")
    
    return render_template('register.html')



@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['useremail']
        password = request.form['password']

        # Query to get the hashed password for the given email
        query = "SELECT password FROM users1 WHERE email = %s"
        values = (email,)
        password_data = retrivequery1(query, values)

        if password_data:
            stored_hashed_password = password_data[0][0]  # Extract stored hash

            # Debugging logs (remove in production)
            print(f"DEBUG: Entered Password: {password}")
            print(f"DEBUG: Stored Hashed Password: {stored_hashed_password}")

            # Verify password using bcrypt
            if bcrypt.hashpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')) == stored_hashed_password.encode('utf-8'):

                session["user_email"] = email  # Set session

                print(f"DEBUG: User '{email}' logged in successfully!")  
                return redirect("/home")  # Redirect to home page
            else:
                print("DEBUG: Password did not match!")
                return render_template('login.html', message="Invalid Password!!")

        print("DEBUG: This email ID does not exist!")  
        return render_template('login.html', message="This email ID does not exist!")

    return render_template('login.html')




# @app.route('/login', methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form['useremail']
#         password = request.form['password']
        
#         query = "SELECT UPPER(email) FROM users"
#         email_data = retrivequery2(query)
#         email_data_list = []
#         for i in email_data:
#             email_data_list.append(i[0])

#         if email.upper() in email_data_list:
#             query = "SELECT password FROM users WHERE email = %s"
#             values = (email,)
#             password__data = retrivequery1(query, values)
#             if password.upper() == password__data[0][0]:
#                 user_email = email
#                 session["user_email"] = user_email

#                 return redirect("/home")
#             return render_template('home.html', message= "Invalid Password!!")
#         return render_template('login.html', message= "This email ID does not exist!")
#     return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template("upload.html", message="No file part")

        myfile = request.files['video']

        if myfile.filename == '':
            return render_template("upload.html", message="No selected file")

        # Acceptable video formats
        accepted_formats = ['avi', 'mp4', 'mkv']
        if not myfile.filename.split('.')[-1].lower() in accepted_formats:
            message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
            return render_template("upload.html", message=message)

        # Save the uploaded video file
        filename = secure_filename(myfile.filename)
        video_path = os.path.join('static/Video/', filename)
        myfile.save(video_path)

        # Process the video and save a single random frame
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame_index = np.random.randint(0, total_frames)

        saved_frame_path = os.path.join('static/Frames/', f"{os.path.splitext(filename)[0]}_frame.jpg")
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame == random_frame_index:
                # Save the frame as an image
                cv2.imwrite(saved_frame_path, frame)
                break

            current_frame += 1

        cap.release()

        # Process the video and predict the class
        frames = load_single_video(video_path)
        features = preprocess_frames(frames)

        # Limit the number of frames to 20 (if necessary)
        features = features[:MAX_SEQ_LENGTH]  # Shape: (20, 2048)
        features = features[None, :, :]  # Shape: (1, 20, 2048) (batch_size, sequence_length, num_features)
        mask = np.ones((1, len(features[0])))  # Mask for all frames (1 = not masked)

        # Predict the label using the model
        prediction = model.predict([features, mask])

        # Class labels for video prediction (Normal, Violence, Weaponized)
        class_labels = ['Normal', 'Violence', 'Weaponized']
        predicted_class = class_labels[np.argmax(prediction)]
        print(f"Predicted class: {predicted_class}")
        if predicted_class != "Normal":
            save_anomaly_to_db( saved_frame_path, predicted_class)

        send_email_with_attachment("Admin email", saved_frame_path)

        # Return the prediction along with the video path to the template
        return render_template('upload.html', prediction=predicted_class, path=video_path)

    return render_template('upload.html')
import mysql.connector
import cv2
import numpy as np




import geocoder

def get_current_location():
    g = geocoder.ip('me')  # Gets the current IP-based location
    if g.ok and g.latlng:  # Check if a valid location is returned
        return f"{g.latlng[0]}, {g.latlng[1]}"
    return "Unknown Location"

# Test if it works
print(get_current_location())  

# def get_full_address():
#     lat, lon = get_current_location()  # Get coordinates
#     if lat is None or lon is None:
#         return "Unknown Location"
    
#     api_key = "YOUR_GOOGLE_API_KEY"  # Replace with your API key
#     url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
    
#     response = requests.get(url).json()
#     if response["status"] == "OK":
#         return response["results"][0]["formatted_address"]  # Return full address
#     return "Unknown Address"

def save_anomaly_to_db( image_path, anomaly_name):
    location = get_current_location()
    
    # Read the image as binary
    with open(image_path, "rb") as file:
        image_data = file.read()

    cursor = mydb.cursor()

    # Insert image into database
    query =  "INSERT INTO anomaly_images (anomaly_name, image, location) VALUES (%s, %s, %s)"
    cursor.execute(query, (anomaly_name, image_data,location))
    mydb.commit()

    print("Anomaly image saved successfully!")
    cursor.close()




# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         if 'video' not in request.files:
#             return render_template("upload.html", message="No file part")

#         myfile = request.files['video']

#         if myfile.filename == '':
#             return render_template("upload.html", message="No selected file")

#         # Acceptable video formats
#         accepted_formats = ['avi', 'mp4', 'mov', 'mkv']
#         if not myfile.filename.split('.')[-1].lower() in accepted_formats:
#             message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
#             return render_template("upload.html", message=message)

#         # Save the uploaded video file
#         filename = secure_filename(myfile.filename)
#         video_path = os.path.join('static/Video/', filename)
#         myfile.save(video_path)

#         # Process the video and predict the class
#         frames = load_single_video(video_path)
#         features = preprocess_frames(frames)

#         # Limit the number of frames to 20 (if necessary)
#         features = features[:MAX_SEQ_LENGTH]  # Shape: (20, 2048)
#         features = features[None, :, :]  # Shape: (1, 20, 2048) (batch_size, sequence_length, num_features)
#         mask = np.ones((1, len(features[0])))  # Mask for all frames (1 = not masked)

#         # Predict the label using the model
#         prediction = model.predict([features, mask])

#         # Class labels for video prediction (Normal, Violence, Weaponized)
#         class_labels = ['Normal', 'Violence', 'Weaponized']
#         predicted_class = class_labels[np.argmax(prediction)]

#         # Save one random frame
#         selected_frame = frames[0]  # Picking the first frame
#         image_path = os.path.join('static/Frames/', 'detected_frame.jpg')
#         cv2.imwrite(image_path, cv2.cvtColor(selected_frame, cv2.COLOR_RGB2BGR))
#         if(predicted_class!="Normal"):
#             return render_template('upload.html', prediction=predicted_class,message="Done")
            


#     return render_template('upload.html')
'''
        # Send email alert
        user_email = session["user_email"]
        owner_email = user_email # Replace with the actual owner email
        email_sent = send_email_with_attachment(owner_email, image_path)

        if email_sent:
            return render_template('upload.html', prediction=predicted_class, message="Alert sent successfully!")
        else:
            return render_template('upload.html', prediction=predicted_class, message="Alert could not be sent. Please try again.")

    return render_template('upload.html')'''

# Function to send an email alert
import smtplib
from email.message import EmailMessage
import imghdr
import os
import cv2
sender_address = "sender email"
sender_pass = " your app password"  
def send_email_with_attachment(owner_email, image_path):
    try:
        new_message = EmailMessage()
        new_message['Subject'] = "Smart Surveilance Alert"
        new_message['From'] = sender_address
        new_message['To'] = "admin mail"
        new_message.set_content('Anomoly action detection. Please check the attached image.')

        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = os.path.basename(f.name)

        new_message.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

        print("Attempting to connect to the SMTP server...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_address, sender_pass)
            smtp.send_message(new_message)
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

    
if __name__ == '__main__':
    app.run(debug = True)