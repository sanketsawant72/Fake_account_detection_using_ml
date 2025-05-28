from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import re
import random
import instaloader

app = Flask(__name__)

# Load model and scaler
try:
    model = load_model('models/instagram_model.h5')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

# Load instagram.csv dataset
try:
    dataset = pd.read_csv('instagram.csv')
    # Select feature columns (exclude 'fake')
    feature_columns = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
                       'name==username', 'description length', 'external URL', 'private',
                       '#posts', '#followers', '#follows']
    dataset_features = dataset[feature_columns]
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Initialize Instaloader
L = instaloader.Instaloader()

def is_valid_instagram_username(username):
    """Validate Instagram username format"""
    pattern = r'^[a-zA-Z0-9._]{1,30}$'
    return bool(re.match(pattern, username))

def get_random_features():
    """Get random row from dataset to simulate profile features"""
    random_row = dataset_features.sample(n=1, random_state=random.randint(0, 10000))
    return random_row

def get_profile_features(username):
    """Fetch Instagram profile features using Instaloader"""
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        
        features = {
            'profile pic': 1 if profile.profile_pic_url else 0,
            'nums/length username': sum(c.isdigit() for c in profile.username) / len(profile.username),
            'fullname words': len(profile.full_name.split()),
            'nums/length fullname': sum(c.isdigit() for c in profile.full_name) / (len(profile.full_name) if profile.full_name else 1),
            'name==username': int(profile.full_name.lower() == profile.username.lower()),
            'description length': len(profile.biography),
            'external URL': 1 if profile.external_url else 0,
            'private': int(profile.is_private),
            '#posts': profile.mediacount,
            '#followers': profile.followers,
            '#follows': profile.followees
        }
        
        features_df = pd.DataFrame([features])
        return features_df
    except Exception as e:
        print(f"Error fetching profile for {username}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form['username'].strip()
    
    try:
        if not is_valid_instagram_username(username):
            raise ValueError("Invalid Instagram username. Use only letters, numbers, periods, or underscores.")
        
        # Try fetching real profile features
        features_df = get_profile_features(username)
        
        if features_df is None:
            features_df = get_random_features()
            note = "Note: Real profile could not be fetched. Using random profile features."
        else:
            note = "Note: Prediction based on real-time fetched profile data."
        
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        confidence = prediction[0][1]
        is_fake = confidence > 0.5
        
        return render_template('result.html', 
                                username=username,
                                is_fake=is_fake,
                                confidence=confidence,
                                note=note)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
