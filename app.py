from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import time
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Initialize app and database
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite DB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load MusicGen model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MusicGen.get_pretrained('facebook/musicgen-medium', device=device)
model.set_generation_params(use_sampling=True, top_k=250, duration=10)

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('generate_audio'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('generate_audio'))
        flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/generate_audio', methods=['GET', 'POST'])
@login_required
def generate_audio():
    if request.method == 'POST':
        description = request.form['description']
        
        output_filename = f'generated_audio_{current_user.id}_{int(time.time())}'  # Removed .wav here
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)  # No .wav extension

        # Generate music
        wav = model.generate([description])
        audio_write(output_path, wav[0], model.sample_rate, format='wav')  # This will add .wav

        # Store the correct filename in session
        session['generated_audio'] = f"{output_filename}.wav"

        return render_template('generate.html', audio_file=session['generated_audio'])

    return render_template('generate.html', audio_file=None)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/download', methods=['GET'])
@login_required
def download():
    file_name = session.get('generated_audio')  # Retrieve the correct filename
    if not file_name:
        flash('No audio file found. Please generate audio first.')
        return redirect(url_for('generate_audio'))
    
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    
    if not os.path.exists(file_path):
        flash('Audio file not found. Please generate audio again.')
        return redirect(url_for('generate_audio'))
    
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    # Initialize the database
    with app.app_context():
        db.create_all()
    app.run(debug=True)