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
from datetime import datetime, timedelta
import json
import re

# Import Signal Processing Suite
from SignalProcessingSuite.blocks import get_block, list_blocks, timed_run

app = Flask(__name__)
app.secret_key = 'leibnitz-super-secret-key-2026'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, 'users.json')

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

# PostgreSQL setup
DATABASE_URL = os.environ.get('DATABASE_URL')

# Try reading from postgreconnection.txt if DATABASE_URL is not set
if not DATABASE_URL:
    conn_file = os.path.join(BASE_DIR, 'postgreconnection.txt')
    if os.path.exists(conn_file):
        try:
            with open(conn_file, 'r') as f:
                DATABASE_URL = f.read().strip()
        except Exception as e:
            print(f"Error reading postgreconnection.txt: {e}")

def init_db():
    if not DATABASE_URL:
        print("DATABASE_URL not set. Running in JSON mode.")
        return
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                is_paid BOOLEAN NOT NULL DEFAULT FALSE,
                payment_utr VARCHAR(32)
            );
        """)
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_paid BOOLEAN NOT NULL DEFAULT FALSE;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS payment_utr VARCHAR(32);")
        # Insert demo user if not exists
        cur.execute("SELECT 1 FROM users WHERE username = 'demo';")
        if not cur.fetchone():
            cur.execute("INSERT INTO users (username, password) VALUES ('demo', 'demo123');")
            print("Default demo user inserted into PostgreSQL database.")
        conn.commit()
        cur.close()
        conn.close()
        print("PostgreSQL database initialized successfully.")
    except Exception as e:
        print(f"Error initializing PostgreSQL database: {e}")



def get_user_password(username):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT password FROM users WHERE username = %s;", (username,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                return row[0]
            return None
        except Exception as e:
            print(f"Database error in get_user_password: {e}. Falling back to JSON.")
    
    # JSON fallback
    users = load_users()
    user_record = users.get(username)
    if isinstance(user_record, dict):
        return user_record.get('password')
    return user_record

def add_user(username, password):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s);", (username, password))
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error in add_user: {e}. Falling back to JSON.")
            
    # JSON fallback
    users = load_users()
    users[username] = {
        "password": password,
        "is_paid": False,
        "payment_utr": None
    }
    save_users(users)
    return True

def is_paid_user(username):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("SELECT is_paid FROM users WHERE username = %s;", (username,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return bool(row[0]) if row else False
        except Exception as e:
            print(f"Database error in is_paid_user: {e}. Falling back to JSON.")

    users = load_users()
    user_record = users.get(username)
    if isinstance(user_record, dict):
        return bool(user_record.get('is_paid', False))
    return False

def mark_user_paid(username, utr=None):
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET is_paid = TRUE, payment_utr = COALESCE(%s, payment_utr) WHERE username = %s;",
                (utr, username)
            )
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error in mark_user_paid: {e}. Falling back to JSON.")

    users = load_users()
    user_record = users.get(username)
    if isinstance(user_record, dict):
        user_record['is_paid'] = True
        if utr:
            user_record['payment_utr'] = utr
    else:
        user_record = {
            "password": user_record,
            "is_paid": True,
            "payment_utr": utr
        }
    users[username] = user_record
    save_users(users)
    return True

def has_unlimited_access():
    return bool(session.get('is_paid')) or is_paid_user(session.get('user'))

# ====================== File Metadata Database (PostgreSQL) ======================

def init_files_table():
    """Initialize the files table in PostgreSQL"""
    if not DATABASE_URL:
        print("DATABASE_URL not set. File tracking requires PostgreSQL.")
        return
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) UNIQUE NOT NULL,
                original_name VARCHAR(255) NOT NULL,
                username VARCHAR(100) NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(100) DEFAULT 'Uploaded',
                processing_count INT DEFAULT 0,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_username ON files(username);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_uploaded_at ON files(uploaded_at DESC);")
        conn.commit()
        cur.close()
        conn.close()
        print("Files table initialized successfully.")
    except Exception as e:
        print(f"Error initializing files table: {e}")

# Initialize Database
init_db()
init_files_table()


def add_file_metadata(filename, original_filename, username):
    """Add metadata for an uploaded file to PostgreSQL"""
    if not DATABASE_URL:
        print("File tracking requires DATABASE_URL")
        return
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO files (filename, original_name, username, status)
            VALUES (%s, %s, %s, 'Uploaded')
            ON CONFLICT (filename) DO NOTHING;
        """, (filename, original_filename, username))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error adding file metadata: {e}")

def get_user_files(username, limit=10):
    """Get recent files for a specific user from PostgreSQL"""
    if not DATABASE_URL:
        return []
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT filename, original_name, uploaded_at, status, processing_count
            FROM files
            WHERE username = %s
            ORDER BY uploaded_at DESC
            LIMIT %s;
        """, (username, limit))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        formatted_files = []
        for filename, original_name, uploaded_at, status, processing_count in rows:
            # Format time difference
            time_diff = datetime.now() - uploaded_at.replace(tzinfo=None)
            
            if time_diff.days > 0:
                time_str = f"{time_diff.days}d ago"
            elif time_diff.seconds > 3600:
                time_str = f"{time_diff.seconds // 3600}h ago"
            elif time_diff.seconds > 60:
                time_str = f"{time_diff.seconds // 60}m ago"
            else:
                time_str = "now"
            
            formatted_files.append({
                'filename': filename,
                'name': original_name,
                'uploaded_at': time_str,
                'status': status or 'Ready',
                'processing_count': processing_count
            })
        
        return formatted_files
    except Exception as e:
        print(f"Error fetching user files: {e}")
        return []

def update_file_status(filename, operation_name):
    """Update file status after processing"""
    if not DATABASE_URL:
        return
    
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            UPDATE files
            SET status = %s, processing_count = processing_count + 1
            WHERE filename = %s;
        """, (f"Processed • {operation_name}", filename))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error updating file status: {e}")

def get_wav_sample_rate(filepath):
    import wave
    try:
        with wave.open(filepath, 'rb') as w:
            return w.getframerate()
    except Exception as e:
        print(f"Error reading WAV header: {e}")
        return None

def detect_csv_properties(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = [f.readline() for _ in range(101)]
        start_idx = 1 if any(c.isalpha() for c in lines[0]) else 0
        data_lines = lines[start_idx:]
        rows = []
        for line in data_lines:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                pass
        if len(rows) < 5:
            return None
        data = np.array(rows)
        if data.ndim == 2 and data.shape[1] > 1:
            col0 = data[:, 0]
            diffs = np.diff(col0)
            if len(diffs) > 0 and np.all(diffs > 0) and np.std(diffs) < 1e-3 * np.mean(diffs):
                mean_diff = np.mean(diffs)
                if mean_diff > 0:
                    return float(round(1.0 / mean_diff, 2))
    except Exception as e:
        print(f"Error detecting CSV properties: {e}")
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_signal_file(filepath, filename):
    detected_rate = None

    if filename.lower().endswith('.wav'):
        from scipy.io import wavfile
        sample_rate_wav, signal_data = wavfile.read(filepath)
        detected_rate = float(sample_rate_wav)

        # Ensure 1D (mono)
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0]

        # Normalize to [-1.0, 1.0] if integer type
        if signal_data.dtype.kind in 'iu':
            max_val = np.iinfo(signal_data.dtype).max
            signal_data = signal_data.astype(np.float32) / max_val

    elif filename.lower().endswith('.npy'):
        signal_data = np.load(filepath)

    elif filename.lower().endswith('.csv'):
        # Load with delimiter
        signal_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        # Detect rate and extract signal column if 2D
        if len(signal_data.shape) > 1 and signal_data.shape[1] > 1:
            col0 = signal_data[:, 0]
            diffs = np.diff(col0)
            if len(diffs) > 0 and np.all(diffs > 0) and np.std(diffs) < 1e-3 * np.mean(diffs):
                mean_diff = np.mean(diffs)
                if mean_diff > 0:
                    detected_rate = float(round(1.0 / mean_diff, 2))
                # Column 0 is time, Column 1 is signal
                signal_data = signal_data[:, 1]
            else:
                signal_data = signal_data[:, 0]
        else:
            signal_data = signal_data.flatten()

    else:  # txt or others
        signal_data = np.loadtxt(filepath)
        if len(signal_data.shape) > 1:
            signal_data = signal_data[:, 0] if signal_data.shape[1] > 1 else signal_data.flatten()

    return np.asarray(signal_data, dtype=float).flatten(), detected_rate

def get_request_sample_rate(data, detected_rate):
    sample_rate = detected_rate if detected_rate is not None else 1000.0
    user_rate = data.get('sample_rate')
    if user_rate is not None:
        try:
            val = float(user_rate)
            if val > 0:
                sample_rate = val
        except (ValueError, TypeError):
            pass
    return sample_rate

def normalize_block_specs(data):
    blocks = data.get('blocks')
    if blocks is not None:
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("blocks must be a non-empty list")
        normalized = []
        for index, spec in enumerate(blocks):
            if not isinstance(spec, dict):
                raise ValueError(f"blocks[{index}] must be an object")
            block_id = spec.get('id') or spec.get('operation')
            if not block_id:
                raise ValueError(f"blocks[{index}] is missing id")
            normalized.append({
                "id": block_id,
                "params": spec.get('params') or {}
            })
        return normalized

    operation = data.get('operation')
    if not operation:
        raise ValueError("Missing operation or blocks")
    legacy_params = {
        key: value
        for key, value in data.items()
        if key not in {'filename', 'operation', 'sample_rate'}
    }
    return [{"id": operation, "params": legacy_params}]

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
                
            if get_user_password(username_val) is not None:
                flash('Username already exists.', 'error')
                return render_template('login.html', active_tab='register', username=username_val)
                
            add_user(username_val, password)
            
            # Autologin and set permanent session
            session.permanent = True
            session['user'] = username_val
            session['is_paid'] = False
            flash('Registration successful! Welcome to Leibnitz.', 'success')
            return redirect(url_for('dashboard'))
            
        else:  # action == 'login'
            password = request.form.get('password')
            db_password = get_user_password(username_val)
            
            if db_password and db_password == password:
                session.permanent = True
                session['user'] = username_val
                session['is_paid'] = is_paid_user(username_val)
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials.', 'error')
                return render_template('login.html', active_tab='login', username=username_val)
                
    return render_template('login.html', active_tab=active_tab, username=username_val)


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('is_paid', None)
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
        "limit": None if has_unlimited_access() else 50,
        "unlimited_access": has_unlimited_access()
    })

@app.route('/api/payment/confirm', methods=['POST'])
def confirm_payment():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    utr = str(data.get('utr', '')).strip()
    if utr and not re.fullmatch(r'\d{12}', utr):
        return jsonify({"error": "Invalid UTR number"}), 400

    mark_user_paid(session['user'], utr or None)
    session['is_paid'] = True
    session['usage_count'] = 0
    return jsonify({
        "success": True,
        "unlimited_access": True,
        "message": "Payment confirmed! Unlimited access unlocked."
    })

@app.route('/api/blocks', methods=['GET'])
def available_blocks():
    return jsonify({"blocks": list_blocks()})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not has_unlimited_access() and session.get('usage_count', 0) >= 50:
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
        
        # Track file metadata for dashboard
        add_file_metadata(filename, file.filename, session.get('user'))
        
        detected_rate = None
        if filename.lower().endswith('.wav'):
            detected_rate = get_wav_sample_rate(filepath)
        elif filename.lower().endswith('.csv'):
            detected_rate = detect_csv_properties(filepath)
            
        if not has_unlimited_access():
            session['usage_count'] = session.get('usage_count', 0) + 1
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "File uploaded successfully",
            "detected_sample_rate": detected_rate,
            "usage_count": session.get('usage_count', 0)
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/recent-files', methods=['GET'])
def get_recent_files():
    """Get recent files for the current user"""
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    username = session.get('user')
    files = get_user_files(username, limit=10)
    
    return jsonify({
        "files": files
    })

@app.route('/api/process', methods=['POST'])
def process_signal():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not has_unlimited_access() and session.get('usage_count', 0) >= 50:
        return jsonify({"error": "Usage limit reached", "redirect": "/payment"}), 402

    data = request.get_json() or {}
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "Missing parameters"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        signal_data, detected_rate = load_signal_file(filepath, filename)
        sample_rate = get_request_sample_rate(data, detected_rate)
        block_specs = normalize_block_specs(data)

        current_signal = signal_data
        pipeline_results = []

        for index, spec in enumerate(block_specs):
            block = get_block(spec["id"])
            params = spec.get("params") or {}
            block_input = current_signal
            run_result, elapsed_ms = timed_run(block, block_input, sample_rate, params)
            plot_path = generate_suite_plot(filename, block.id, block_input, run_result.plot_data, sample_rate=sample_rate)

            pipeline_results.append({
                "index": index,
                "id": block.id,
                "name": block.name,
                "category": block.category,
                "params": params,
                "result": run_result.result,
                "metadata": run_result.metadata,
                "warnings": run_result.warnings,
                "elapsed_ms": round(elapsed_ms, 3),
                "plot_url": f"/processed/{plot_path}" if plot_path else None,
            })

            if run_result.output_signal is not None:
                current_signal = np.asarray(run_result.output_signal, dtype=float).flatten()

        first_result = pipeline_results[0] if pipeline_results else {}
        last_result = pipeline_results[-1] if pipeline_results else {}

        # Update file status in PostgreSQL
        if first_result.get('name'):
            update_file_status(filename, first_result.get('name'))

        # Increment usage count only for users still on the free tier.
        if not has_unlimited_access():
            session['usage_count'] = session.get('usage_count', 0) + 1

        output_id = str(uuid.uuid4())
        return jsonify({
            "success": True,
            "operation": first_result.get("id"),
            "result": last_result.get("result"),
            "plot_url": last_result.get("plot_url"),
            "pipeline": pipeline_results,
            "sample_rate": sample_rate,
            "download_url": f"/api/download/{output_id}",
            "usage_count": session.get('usage_count', 0)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_suite_plot(original_filename, operation, original_signal, result_data, sample_rate=1000.0):
    from SignalProcessingSuite.visualization import plot_time, plot_frequency, plot_ifft, plot_wavelet_coefficients
    
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
        
        plot_time(original_signal[:1000], sample_rate=sample_rate, ax=ax1, title="Time Domain Signal")
        ax1.get_lines()[0].set_color('#00ff88')
        
        plot_frequency(original_signal, sample_rate=sample_rate, ax=ax2, db=False, title="Frequency Spectrum (FFT)")
        ax2.get_lines()[0].set_color('#00d4ff')
        ax2.set_xlim(0, sample_rate / 2.0)

    elif operation == 'ifft':
        fig.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='#0a0e27')
        ax1.set_facecolor('#0a0e27')
        ax2.set_facecolor('#0a0e27')
        
       
        
        plot_frequency(original_signal, sample_rate=sample_rate, ax=ax1, db=False, title="Frequency Spectrum (FFT)")
        ax1.get_lines()[0].set_color('#00d4ff')
        ax1.set_xlim(0, sample_rate / 2.0)

        plot_ifft(
            signal,
            sample_rate=sample_rate,
            ax=ax2,
            db=False,
            title="Time Domain IFFT"
        )

        lines = ax2.get_lines()

        if lines:
            lines[0].set_color('#00d4ff')

        ax2.set_xlim(0, 1000)
        # plot_ifft(signal, sample_rate=sample_rate, ax=ax2, db= False, title= "Time Domain IFFT")
        # ax2.get_lines()[0].set_color('#00d4ff')
        # ax2.set_xlim(0, 1000)



    elif operation == 'filter':
        times = np.arange(min(len(original_signal), 1000)) / sample_rate
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
        times = np.arange(min(len(original_signal), 1000)) / sample_rate
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
