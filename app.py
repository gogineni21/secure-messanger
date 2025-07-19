# app.py (Corrected version for data consistency)
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import os
from datetime import datetime
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64
import pickle
import geopy.distance
from hashlib import sha256

app = Flask(__name__)
app.secret_key = "a_very_secret_key_change_me"

# --- File Paths ---
USER_FILE = "users.csv"
ACCESS_MODEL_FILE = "rfc_access_model.pkl"
CONTENT_MODEL_FILE = "content_model.pkl"
MESSAGE_FILE = "messages.csv"

# --- Load ML Models ---
try:
    with open(ACCESS_MODEL_FILE, "rb") as f:
        access_model_data = pickle.load(f)
        access_model = access_model_data['model']
        access_model_columns = access_model_data['columns']
except FileNotFoundError:
    access_model, access_model_columns = None, []
    print(f"Warning: Access model '{ACCESS_MODEL_FILE}' not found.")

try:
    with open(CONTENT_MODEL_FILE, "rb") as f:
        content_model = pickle.load(f)
except FileNotFoundError:
    content_model = None
    print(f"Warning: Content model '{CONTENT_MODEL_FILE}' not found.")

# === Encryption/Decryption ===
def get_aes_key_from_passphrase(passphrase):
    return sha256(passphrase.encode()).digest()

def encrypt_message_aes(key, message):
    cipher = AES.new(key, AES.MODE_CBC)
    return base64.b64encode(cipher.iv + cipher.encrypt(pad(message.encode('utf-8'), AES.block_size))).decode('utf-8')

def decrypt_message_aes(key, encrypted_message):
    try:
        data = base64.b64decode(encrypted_message)
        iv, encrypted = data[:AES.block_size], data[AES.block_size:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(encrypted), AES.block_size).decode('utf-8')
    except (ValueError, KeyError):
        raise ValueError("Decryption failed. Incorrect passphrase.")

# === Geolocation ===
def is_within_location(current, target, radius_km):
    if not all(isinstance(c, (int, float)) for c in current) or not all(isinstance(c, (int, float)) for c in target):
        return False
    return geopy.distance.distance(current, target).km <= radius_km

# === User Management ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_user_file():
    # FIX: Standardized on 'user_role'
    if not os.path.exists(USER_FILE):
        pd.DataFrame(columns=["username", "password", "user_role", "department", "last_login_time"]).to_csv(USER_FILE, index=False)

def get_user_details(username):
    users_df = pd.read_csv(USER_FILE)
    user_data = users_df[users_df.username == username]
    return user_data.iloc[0] if not user_data.empty else None

def verify_user(username, password):
    user_data = get_user_details(username)
    if user_data is not None and user_data['password'] == hash_password(password):
        users_df = pd.read_csv(USER_FILE)
        users_df.loc[users_df.username == username, 'last_login_time'] = datetime.now().isoformat()
        users_df.to_csv(USER_FILE, index=False)
        return True
    return False

def register_user(username, password, user_role, department):
    # FIX: Standardized on 'user_role'
    init_user_file()
    users_df = pd.read_csv(USER_FILE)
    if username in users_df.username.values:
        return False
    new_user = pd.DataFrame([[username, hash_password(password), user_role, department, None]], columns=["username", "password", "user_role", "department", "last_login_time"])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USER_FILE, index=False)
    return True

# === Message Management ===
def init_message_file():
    if not os.path.exists(MESSAGE_FILE):
        pd.DataFrame(columns=["sender", "recipient", "receiver_location", "radius", "encrypted_message"]).to_csv(MESSAGE_FILE, index=False)

def save_encrypted_message(sender, recipient, receiver_location, radius, encrypted):
    init_message_file()
    df = pd.read_csv(MESSAGE_FILE)
    new_row = {"sender": sender, "recipient": recipient, "receiver_location": receiver_location, "radius": radius, "encrypted_message": encrypted}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(MESSAGE_FILE, index=False)

# === ML Feature Preparation ===
def prepare_features_for_model(user_data):
    if not access_model: return None
    last_login_dt = pd.to_datetime(user_data.get('last_login_time'))
    # FIX: Standardized on 'user_role'
    features = {
        'access_time_hour': datetime.now().hour,
        'last_login_time_hour': last_login_dt.hour if pd.notnull(last_login_dt) else datetime.now().hour,
        'user_role': user_data['user_role'],
        'department': user_data['department']
    }
    features_df = pd.DataFrame([features])
    features_processed = pd.get_dummies(features_df)
    model_features = pd.DataFrame(columns=access_model_columns)
    final_features = pd.concat([model_features, features_processed]).fillna(0)
    return final_features[access_model_columns]

# === Routes ===
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if verify_user(request.form['username'], request.form['password']):
            session['username'] = request.form['username']
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        flash("Invalid username or password.", "danger")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # FIX: Standardized on 'user_role'
        if register_user(request.form['username'], request.form['password'], request.form['user_role'], request.form['department']):
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('login'))
        flash("Username already exists.", "danger")
    # FIX: Changed template name to match convention
    return render_template('signup.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        message_text = request.form['message']
        if content_model and content_model.predict([message_text])[0] == 1:
            flash("Your message was blocked for containing suspicious content.", "danger")
            return render_template('home.html')
        try:
            key = get_aes_key_from_passphrase(request.form['passphrase'])
            encrypted = encrypt_message_aes(key, message_text)
            save_encrypted_message(
                sender=session['username'], recipient=request.form['recipient'],
                receiver_location=f"{request.form['receiver_lat']},{request.form['receiver_lng']}",
                radius=float(request.form['radius']), encrypted=encrypted )
            flash("Message sent successfully!", "success")
        except Exception as e:
            flash(f"An error occurred: {e}", "danger")
    return render_template('home.html')

@app.route('/view', methods=['GET', 'POST'])
def view():
    if 'username' not in session: return redirect(url_for('login'))
    messages = []
    if request.method == 'POST':
        user_data = get_user_details(session['username'])
        if not user_data.get('user_role'):
            flash("Your user profile is incomplete. Please sign up again.", "danger")
            return render_template('view_messages.html', messages=[])
        
        features = prepare_features_for_model(user_data)
        if access_model and features is not None and access_model.predict(features)[0] == 0:
           flash("Security Alert: Access blocked by intelligent control system.", "warning")
           return render_template('view_messages.html', messages=[])
        
        try:
            current_location = (float(request.form['lat']), float(request.form['lng']))
            key = get_aes_key_from_passphrase(request.form['passphrase'])
        except ValueError:
            flash("Invalid coordinates or passphrase.", "danger")
            return render_template('view_messages.html', messages=[])

        if os.path.exists(MESSAGE_FILE):
            df = pd.read_csv(MESSAGE_FILE).dropna(subset=['recipient', 'radius'])
            user_messages = df[df.recipient == session['username']]
            for _, row in user_messages.iterrows():
                if is_within_location(current_location, tuple(map(float, row['receiver_location'].split(','))), float(row['radius'])):
                    try:
                        messages.append({'sender': row['sender'], 'content': decrypt_message_aes(key, row['encrypted_message'])})
                    except ValueError:
                        flash("Incorrect passphrase for one or more messages.", "danger")
                        break
    return render_template('view_messages.html', messages=messages)

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_user_file()
    init_message_file()
    app.run(debug=True)