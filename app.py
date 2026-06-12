# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:21:02 2026

@author: acer
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session, flash
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import uuid
from datetime import datetime
import json

# Import Signal Processing Suite
from SignalProcessingSuite import magnitude_spectrum, filter_signal, dwt

app = Flask(__name__)
app.secret_key = 'leibnitz-super-secret-key-2026'

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'npy', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Demo user and database setup
DEMO_USER = {"username": "demo", "password": "demo123"}
USERS_FILE = 'users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        users = {DEMO_USER['username']: DEMO_USER['password']}
        save_users(users)
        return users
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
        return {DEMO_USER['username']: DEMO_USER['password']}

def save_users(users):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====================== ROUTES ======================

@app.route('/')
def index():
    return render_template('leibnitz.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
        
    active_tab = 'register'
    username_val = ''
    
    if request.method == 'POST':
        action = request.form.get('action', 'login')
        username_val = request.form.get('username', '').strip()
        
        if action == 'register':
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if not username_val or not password:
                flash('Username and password are required.', 'error')
                return render_template('login.html', active_tab='register', username=username_val)
                
            if password != confirm_password:
                flash('Passwords do not match.', 'error')
                return render_template('login.html', active_tab='register', username=username_val)
                
            users = load_users()
            if username_val in users:
                flash('Username already exists.', 'error')
                return render_template('login.html', active_tab='register', username=username_val)
                
            users[username_val] = password
            save_users(users)
            flash('Registration successful! Please login.', 'success')
            return render_template('login.html', active_tab='login', username=username_val)
            
        else:  # action == 'login'
            password = request.form.get('password')
            users = load_users()
            
            if username_val in users and users[username_val] == password:
                session['user'] = username_val
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials.', 'error')
                return render_template('login.html', active_tab='login', username=username_val)
                
    return render_template('login.html', active_tab=active_tab, username=username_val)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/payment')
def payment_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('payment.html', usage_count=session.get('usage_count', 0))

# ====================== API ENDPOINTS ======================

@app.route('/api/usage')
def get_usage():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({
        "usage_count": session.get('usage_count', 0),
        "limit": 5
    })

@app.route('/api/payment/confirm', methods=['POST'])
def confirm_payment():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    session['usage_count'] = 0
    return jsonify({
        "success": True,
        "message": "Payment confirmed! Free usage limit reset."
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if session.get('usage_count', 0) >= 5:
        return jsonify({"error": "Usage limit reached", "redirect": "/payment"}), 402

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        session['usage_count'] = session.get('usage_count', 0) + 1
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "File uploaded successfully",
            "usage_count": session.get('usage_count', 0)
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/process', methods=['POST'])
def process_signal():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if session.get('usage_count', 0) >= 5:
        return jsonify({"error": "Usage limit reached", "redirect": "/payment"}), 402

    data = request.get_json()
    filename = data.get('filename')
    operation = data.get('operation')

    if not filename or not operation:
        return jsonify({"error": "Missing parameters"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        # Load signal data
        if filename.endswith('.npy'):
            signal_data = np.load(filepath)
        elif filename.endswith('.csv'):
            signal_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        else:  # txt or others
            signal_data = np.loadtxt(filepath)

        # Ensure 1D
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0] if signal_data.shape[1] > 1 else signal_data.flatten()

        result = {}
        plot_path = None
        result_data_for_plot = {}

        if operation == 'fft':
            # Use SignalProcessingSuite to compute magnitude spectrum
            xf, magnitude = magnitude_spectrum(signal_data, sample_rate=1000.0, window=None)
            
            result = {
                "frequencies": xf.tolist()[:500],
                "magnitude": magnitude.tolist()[:500]
            }
            # Generate plot
            plot_path = generate_suite_plot(filename, 'fft', signal_data, result)

        elif operation == 'filter':
            # Design and apply filter using SignalProcessingSuite
            filtered = filter_signal(signal_data, sample_rate=1000.0, cutoff=100.0, kind='lowpass', method='butter')

            result = {"filtered": filtered.tolist()[:1000]}
            result_data_for_plot = {'filtered': filtered}
            plot_path = generate_suite_plot(filename, 'filter', signal_data, result_data_for_plot)

        elif operation == 'wavelet':
            # Perform multilevel Haar/DWT from SignalProcessingSuite
            coeffs = dwt(signal_data, wavelet='db4', levels=3)
            
            # Save raw objects for plotting
            result_data_for_plot = {'_coeffs_obj': coeffs}
            
            # Serialize summary stats
            result = {
                "wavelet": "db4",
                "levels": len(coeffs) - 1,
                "coefficients_summary": [
                    {
                        "band": "Approximation" if i == 0 else f"Detail Level {len(coeffs) - i}",
                        "size": len(c),
                        "mean": float(np.mean(c)),
                        "std": float(np.std(c)),
                        "max": float(np.max(c)),
                        "min": float(np.min(c))
                    }
                    for i, c in enumerate(coeffs)
                ]
            }
            plot_path = generate_suite_plot(filename, 'wavelet', signal_data, result_data_for_plot)

        # Increment usage count
        session['usage_count'] = session.get('usage_count', 0) + 1

        output_id = str(uuid.uuid4())
        return jsonify({
            "success": True,
            "operation": operation,
            "result": result,
            "plot_url": f"/processed/{plot_path}" if plot_path else None,
            "download_url": f"/api/download/{output_id}",
            "usage_count": session.get('usage_count', 0)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_suite_plot(original_filename, operation, original_signal, result_data):
    from SignalProcessingSuite.visualization import plot_time, plot_frequency, plot_wavelet_coefficients
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
    ax.set_facecolor('#0a0e27')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{operation}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(app.config['PROCESSED_FOLDER'], plot_filename)

    if operation == 'fft':
        fig.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='#0a0e27')
        ax1.set_facecolor('#0a0e27')
        ax2.set_facecolor('#0a0e27')
        
        plot_time(original_signal[:1000], sample_rate=1000.0, ax=ax1, title="Time Domain Signal")
        ax1.get_lines()[0].set_color('#00ff88')
        
        plot_frequency(original_signal, sample_rate=1000.0, ax=ax2, db=False, title="Frequency Spectrum (FFT)")
        ax2.get_lines()[0].set_color('#00d4ff')
        ax2.set_xlim(0, 500)

    elif operation == 'filter':
        times = np.arange(min(len(original_signal), 1000)) / 1000.0
        ax.plot(times, original_signal[:1000], label='Original', alpha=0.7, color='#ff3344')
        ax.plot(times, result_data['filtered'][:1000], label='Filtered', color='#00ff88')
        ax.set_title('Low-Pass Filtered Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif operation == 'wavelet':
        fig.clf()
        fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
        ax1.set_facecolor('#0a0e27')
        
        coeffs = result_data['_coeffs_obj']
        plot_wavelet_coefficients(coeffs, ax=ax1)
        ax1.grid(True, alpha=0.3)

    else:
        times = np.arange(min(len(original_signal), 1000)) / 1000.0
        ax.plot(times, original_signal[:1000], color='#ffd700')
        ax.set_title('Signal Visualization')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, facecolor='#0a0e27')
    plt.close('all')
    
    return plot_filename

@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/api/download/<output_id>')
def download_output(output_id):
    # For now, return a sample processed file
    # Can be extended to zip multiple outputs
    return jsonify({"message": "Download feature - ready for extension"})

if __name__ == '__main__':
    print("[Leibnitz Signal Processing Backend Started]")
    print("Demo Login: username: demo | password: demo123")
    app.run(debug=True, host='0.0.0.0', port=5000)