"""
Smart Attendance Management System for Company Employees
Enterprise Full-Stack Flask Web Application
Features:
- Employee registration with biometric face enrollment and unique QR generation
- Real-time webcam Face Recognition attendance via OpenCV YuNet (detection) and SFace (recognition)
- Real-time camera QR scanning via unguessable revocable cryptographic tokens
- Complete Check-In & Check-Out workflow with shift rules, grace periods, and overnight handling
- Comprehensive dashboards, employee directory, audit logging, analytics, and CSV exports
- Role-based admin authentication with rate-limiting, CSRF protection, and SQLite parameterization
"""

import os
import io
import time
import csv
import base64
import secrets
from datetime import datetime, date, timedelta
from functools import wraps

import cv2
import numpy as np
import qrcode
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, Response, abort, send_from_directory
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# Local modules
from db import (
    get_db_connection, init_db, generate_secure_qr_token, hash_qr_token, BASE_DIR,
    VALID_DEPARTMENTS, VALID_PRODUCTION_BLOCKS
)
from models_utils import FaceEngine, COSINE_THRESHOLD

app = Flask(__name__)

# Security & Session Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'smart-attendance-enterprise-secret-key-2026')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file upload limit

# Directories
STATIC_DIR = os.path.join(BASE_DIR, 'static')

if os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
    PHOTOS_DIR = os.path.join('/tmp', 'images', 'employees')
    QR_CODES_DIR = os.path.join('/tmp', 'qr_codes')
    FACE_DATA_DIR = os.path.join('/tmp', 'face_data')
else:
    PHOTOS_DIR = os.path.join(STATIC_DIR, 'images', 'employees')
    QR_CODES_DIR = os.path.join(BASE_DIR, 'qr_codes')
    FACE_DATA_DIR = os.path.join(BASE_DIR, 'face_data')

try:
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    os.makedirs(QR_CODES_DIR, exist_ok=True)
    os.makedirs(FACE_DATA_DIR, exist_ok=True)
except Exception:
    pass

# Login rate-limiting storage: { ip: [timestamp, ...] }
FAILED_LOGIN_ATTEMPTS = {}
LOCKOUT_DURATION = 15 * 60  # 15 minutes
MAX_FAILED_ATTEMPTS = 5

# Initialize database schema and Face Engine singleton
init_db()
face_engine = FaceEngine.get_instance()


# ==============================================================================
# Security & CSRF Helpers
# ==============================================================================

def generate_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf_token())

@app.before_request
def csrf_protect():
    if request.method == "POST":
        # Check CSRF token for state-changing POST requests
        token = request.headers.get("X-CSRFToken") or request.form.get("csrf_token")
        expected_token = session.get("csrf_token")
        
        # Whitelist API endpoints if using valid token check or verify token
        if not expected_token or token != expected_token:
            # Check JSON body for token as well
            if request.is_json:
                json_token = request.json.get("csrf_token") if request.json else None
                if json_token and json_token == expected_token:
                    return
            # If token mismatch
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "error": "Invalid or expired security token (CSRF)."}), 403
            flash("Security validation failed. Please refresh the page and try again.", "danger")
            return redirect(request.referrer or url_for("dashboard"))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access the enterprise portal.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


# ==============================================================================
# Helper Business Logic: Shift & Attendance Calculations
# ==============================================================================

def parse_time_to_minutes(time_str: str) -> int:
    """Converts 'HH:MM' or 'HH:MM:SS' string into minutes from midnight."""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    return hours * 60 + minutes

def calculate_session_duration(check_in_str: str, check_out_str: str, is_overnight: bool = False) -> int:
    """Calculates total working minutes between check-in and check-out timestamps."""
    fmt = "%H:%M:%S"
    try:
        t_in = datetime.strptime(check_in_str, fmt)
        t_out = datetime.strptime(check_out_str, fmt)
        
        if is_overnight or t_out < t_in:
            # Checkout occurred next calendar day
            diff = (t_out + timedelta(days=1)) - t_in
        else:
            diff = t_out - t_in
            
        return max(int(diff.total_seconds() // 60), 0)
    except Exception as e:
        print(f"Error calculating duration: {e}")
        return 0

def process_attendance_event(employee_id: str, method: str, event_type: str = 'auto') -> dict:
    """
    Comprehensive daily workforce attendance state-machine:
    - Shift Check-In (Work Begins)
    - Tea Break (Out & In)
    - Lunch Break (Out & In)
    - Shift Close / Check-Out (Work Day Ends with Net Effective Productive Hours)
    - Duplicate detection & automatic camera shut off triggers
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Verify active employee
    cursor.execute("""
        SELECT e.*, s.shift_name, s.start_time, s.end_time, s.grace_period_minutes 
        FROM employees e
        LEFT JOIN shifts s ON e.shift_id = s.id
        WHERE e.employee_id = ?
    """, (employee_id,))
    employee = cursor.fetchone()

    if not employee:
        conn.close()
        return {"success": False, "error": f"Employee {employee_id} not found."}

    if employee['status'] != 'active':
        conn.close()
        return {"success": False, "error": f"Employee {employee['full_name']} is deactivated and cannot record attendance."}

    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Shift settings
    shift_start = employee['start_time'] or "09:00"
    shift_end = employee['end_time'] or "17:00"
    grace_period = employee['grace_period_minutes'] if employee['grace_period_minutes'] is not None else 15
    is_overnight = parse_time_to_minutes(shift_start) > parse_time_to_minutes(shift_end)

    # 2. Query today's attendance record
    cursor.execute("SELECT * FROM attendance WHERE employee_id = ? AND attendance_date = ?", (employee_id, today_date))
    record = cursor.fetchone()

    # --------------------------------------------------------------------------
    # CASE A: NO RECORD TODAY -> SHIFT CHECK-IN
    # --------------------------------------------------------------------------
    if not record:
        if event_type in ['tea_break', 'lunch_break', 'check_out', 'shift_close']:
            conn.close()
            return {
                "success": False,
                "error": f"{employee['full_name']} must record Shift Check-In before logging breaks or check-out."
            }

        # Check punctuality
        current_mins = parse_time_to_minutes(current_time)
        shift_start_mins = parse_time_to_minutes(shift_start)
        is_late = 1 if current_mins > (shift_start_mins + grace_period) else 0
        status = 'Late' if is_late else 'Present'

        cursor.execute("""
            INSERT INTO attendance 
            (employee_id, attendance_date, check_in_time, check_out_time, total_working_minutes,
             effective_working_minutes, current_activity, attendance_method, status, is_late, is_early_departure, created_at, updated_at)
            VALUES (?, ?, ?, NULL, 0, 0, 'On-Duty', ?, ?, ?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (employee_id, today_date, current_time, method, status, is_late))
        conn.commit()
        conn.close()

        late_msg = f" (Late arrival past {shift_start} + {grace_period}m grace)" if is_late else " (On Time)"
        return {
            "success": True,
            "action": "check_in",
            "employee": dict(employee),
            "attendance": {
                "action": "check_in",
                "attendance_date": today_date,
                "event_time": current_time,
                "check_in_time": current_time,
                "current_activity": "On-Duty",
                "is_late": bool(is_late),
                "status": status,
                "method": method
            },
            "message": f"Shift Check-In recorded at {current_time}{late_msg}. Welcome to Himalaya Wellness, {employee['full_name']}!"
        }

    # --------------------------------------------------------------------------
    # CASE B: SHIFT ALREADY CLOSED FOR TODAY
    # --------------------------------------------------------------------------
    if record['check_out_time']:
        conn.close()
        gross_mins = record['total_working_minutes'] or 0
        net_mins = record['effective_working_minutes'] or gross_mins
        breaks = record['total_break_minutes'] or 0
        time_str = f"Net: {net_mins // 60}h {net_mins % 60}m (Breaks: {breaks}m)"
        return {
            "success": False,
            "is_completed": True,
            "error": f"Shift already closed today for {employee['full_name']} (Checked out at {record['check_out_time']}. {time_str})."
        }

    # --------------------------------------------------------------------------
    # CASE C: ACTIVE SHIFT -> TEA BREAK
    # --------------------------------------------------------------------------
    current_activity = record['current_activity'] or 'On-Duty'
    
    if event_type == 'tea_break' or (event_type == 'auto' and current_activity == 'Tea Break'):
        if current_activity == 'Tea Break':
            # Resuming from Tea Break (Tea Break In)
            start_time = record['tea_break_start'] or current_time
            diff_mins = calculate_session_duration(start_time, current_time)
            new_tea_mins = (record['tea_break_minutes'] or 0) + diff_mins
            total_breaks = new_tea_mins + (record['lunch_break_minutes'] or 0)

            cursor.execute("""
                UPDATE attendance
                SET tea_break_end = ?,
                    tea_break_minutes = ?,
                    total_break_minutes = ?,
                    current_activity = 'On-Duty',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (current_time, new_tea_mins, total_breaks, record['id']))

            # Log to break_logs
            cursor.execute("""
                INSERT INTO break_logs (employee_id, attendance_id, attendance_date, break_type, start_time, end_time, duration_minutes)
                VALUES (?, ?, ?, 'Tea Break', ?, ?, ?)
            """, (employee_id, record['id'], today_date, start_time, current_time, diff_mins))
            conn.commit()
            conn.close()

            return {
                "success": True,
                "action": "tea_break_end",
                "employee": dict(employee),
                "attendance": {
                    "action": "tea_break_end",
                    "attendance_date": today_date,
                    "event_time": current_time,
                    "current_activity": "On-Duty",
                    "break_type": "Tea Break",
                    "break_duration": f"{diff_mins}m",
                    "status": "Resumed Duty",
                    "method": method
                },
                "message": f"Tea Break finished at {current_time} ({diff_mins} mins). Welcome back on duty, {employee['full_name']}!"
            }
        else:
            # Starting Tea Break (Tea Break Out)
            cursor.execute("""
                UPDATE attendance
                SET tea_break_start = ?,
                    current_activity = 'Tea Break',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (current_time, record['id']))
            conn.commit()
            conn.close()

            return {
                "success": True,
                "action": "tea_break_start",
                "employee": dict(employee),
                "attendance": {
                    "action": "tea_break_start",
                    "attendance_date": today_date,
                    "event_time": current_time,
                    "current_activity": "Tea Break",
                    "break_type": "Tea Break",
                    "status": "On Tea Break",
                    "method": method
                },
                "message": f"Tea Break started at {current_time}. Enjoy your tea break, {employee['full_name']}!"
            }

    # --------------------------------------------------------------------------
    # CASE D: ACTIVE SHIFT -> LUNCH BREAK
    # --------------------------------------------------------------------------
    if event_type == 'lunch_break' or (event_type == 'auto' and current_activity == 'Lunch Break'):
        if current_activity == 'Lunch Break':
            # Resuming from Lunch Break (Lunch In)
            start_time = record['lunch_break_start'] or current_time
            diff_mins = calculate_session_duration(start_time, current_time)
            new_lunch_mins = (record['lunch_break_minutes'] or 0) + diff_mins
            total_breaks = (record['tea_break_minutes'] or 0) + new_lunch_mins

            cursor.execute("""
                UPDATE attendance
                SET lunch_break_end = ?,
                    lunch_break_minutes = ?,
                    total_break_minutes = ?,
                    current_activity = 'On-Duty',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (current_time, new_lunch_mins, total_breaks, record['id']))

            # Log to break_logs
            cursor.execute("""
                INSERT INTO break_logs (employee_id, attendance_id, attendance_date, break_type, start_time, end_time, duration_minutes)
                VALUES (?, ?, ?, 'Lunch Break', ?, ?, ?)
            """, (employee_id, record['id'], today_date, start_time, current_time, diff_mins))
            conn.commit()
            conn.close()

            return {
                "success": True,
                "action": "lunch_break_end",
                "employee": dict(employee),
                "attendance": {
                    "action": "lunch_break_end",
                    "attendance_date": today_date,
                    "event_time": current_time,
                    "current_activity": "On-Duty",
                    "break_type": "Lunch Break",
                    "break_duration": f"{diff_mins}m",
                    "status": "Resumed Duty",
                    "method": method
                },
                "message": f"Lunch Break finished at {current_time} ({diff_mins} mins). Welcome back on duty, {employee['full_name']}!"
            }
        else:
            # Starting Lunch Break (Lunch Out)
            cursor.execute("""
                UPDATE attendance
                SET lunch_break_start = ?,
                    current_activity = 'Lunch Break',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (current_time, record['id']))
            conn.commit()
            conn.close()

            return {
                "success": True,
                "action": "lunch_break_start",
                "employee": dict(employee),
                "attendance": {
                    "action": "lunch_break_start",
                    "attendance_date": today_date,
                    "event_time": current_time,
                    "current_activity": "Lunch Break",
                    "break_type": "Lunch Break",
                    "status": "On Lunch Break",
                    "method": method
                },
                "message": f"Lunch Break started at {current_time}. Enjoy your lunch break, {employee['full_name']}!"
            }

    # --------------------------------------------------------------------------
    # CASE E: FINAL SHIFT CLOSE / CHECK-OUT
    # --------------------------------------------------------------------------
    # Calculate gross elapsed duration
    gross_minutes = calculate_session_duration(record['check_in_time'], current_time, is_overnight)
    total_break_mins = record['total_break_minutes'] or 0

    # Auto-close open breaks if any
    if current_activity == 'Tea Break' and record['tea_break_start']:
        add_b = calculate_session_duration(record['tea_break_start'], current_time)
        total_break_mins += add_b
    elif current_activity == 'Lunch Break' and record['lunch_break_start']:
        add_b = calculate_session_duration(record['lunch_break_start'], current_time)
        total_break_mins += add_b

    # Net productive working minutes
    effective_mins = max(gross_minutes - total_break_mins, 0)

    # Check early departure vs shift end
    curr_mins = parse_time_to_minutes(current_time)
    shift_end_mins = parse_time_to_minutes(shift_end)
    is_early_departure = 1 if (not is_overnight and curr_mins < shift_end_mins) else 0

    # Determine status
    if effective_mins < 240:
        status = 'Half-Day'
    elif record['is_late']:
        status = 'Late'
    else:
        status = 'Present'

    cursor.execute("""
        UPDATE attendance
        SET check_out_time = ?,
            total_working_minutes = ?,
            effective_working_minutes = ?,
            total_break_minutes = ?,
            current_activity = 'Shift Closed',
            status = ?,
            is_early_departure = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (current_time, gross_minutes, effective_mins, total_break_mins, status, is_early_departure, record['id']))
    conn.commit()
    conn.close()

    net_formatted = f"{effective_mins // 60}h {effective_mins % 60}m"
    gross_formatted = f"{gross_minutes // 60}h {gross_minutes % 60}m"
    break_formatted = f"{total_break_mins}m"

    return {
        "success": True,
        "action": "check_out",
        "employee": dict(employee),
        "attendance": {
            "action": "check_out",
            "attendance_date": today_date,
            "event_time": current_time,
            "check_in_time": record['check_in_time'],
            "check_out_time": current_time,
            "total_working_time": net_formatted,
            "gross_time": gross_formatted,
            "total_break_time": break_formatted,
            "current_activity": "Shift Closed",
            "is_late": bool(record['is_late']),
            "is_early_departure": bool(is_early_departure),
            "status": status,
            "method": method
        },
        "message": f"Shift Closed at {current_time}. Net Productive Work: {net_formatted} (Breaks: {break_formatted}). Good evening, {employee['full_name']}!"
    }


# ==============================================================================
# Authentication Routes
# ==============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    client_ip = request.remote_addr or '127.0.0.1'
    now_ts = time.time()

    # Rate limiting check
    if client_ip in FAILED_LOGIN_ATTEMPTS:
        # Purge older attempts
        FAILED_LOGIN_ATTEMPTS[client_ip] = [t for t in FAILED_LOGIN_ATTEMPTS[client_ip] if now_ts - t < LOCKOUT_DURATION]
        if len(FAILED_LOGIN_ATTEMPTS[client_ip]) >= MAX_FAILED_ATTEMPTS:
            remaining_mins = int((LOCKOUT_DURATION - (now_ts - FAILED_LOGIN_ATTEMPTS[client_ip][0])) // 60) + 1
            flash(f"Too many failed login attempts. Account temporarily locked. Please try again in {remaining_mins} minute(s).", "danger")
            return render_template('login.html')

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template('login.html')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_users WHERE username = ? AND is_active = 1", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password_hash'], password):
            # Reset failed attempts
            if client_ip in FAILED_LOGIN_ATTEMPTS:
                del FAILED_LOGIN_ATTEMPTS[client_ip]

            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['role'] = user['role']

            cursor.execute("UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user['id'],))
            conn.commit()
            conn.close()

            flash(f"Welcome back, {user['full_name']}!", "success")
            next_url = request.args.get('next')
            return redirect(next_url or url_for('dashboard'))
        else:
            conn.close()
            # Record failed attempt
            if client_ip not in FAILED_LOGIN_ATTEMPTS:
                FAILED_LOGIN_ATTEMPTS[client_ip] = []
            FAILED_LOGIN_ATTEMPTS[client_ip].append(now_ts)
            
            remaining = MAX_FAILED_ATTEMPTS - len(FAILED_LOGIN_ATTEMPTS[client_ip])
            if remaining > 0:
                flash(f"Invalid username or password. {remaining} attempt(s) remaining before temporary lockout.", "danger")
            else:
                flash(f"Invalid username or password. Account locked for 15 minutes.", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have successfully signed out.", "info")
    return redirect(url_for('login'))


# ==============================================================================
# Dashboard View
# ==============================================================================

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()

    today_date = date.today().strftime("%Y-%m-%d")

    # 1. Total Registered Employees
    cursor.execute("SELECT COUNT(*) as total FROM employees WHERE status = 'active'")
    total_employees = cursor.fetchone()['total']

    # 2. Employees Present Today (check-in exists)
    cursor.execute("SELECT COUNT(*) as present FROM attendance WHERE attendance_date = ?", (today_date,))
    present_today = cursor.fetchone()['present']

    # 3. Absent Today
    absent_today = max(total_employees - present_today, 0)

    # 4. Currently Checked-In (checked in but checkout is null)
    cursor.execute("SELECT COUNT(*) as active FROM attendance WHERE attendance_date = ? AND check_out_time IS NULL", (today_date,))
    currently_checked_in = cursor.fetchone()['active']

    # 5. Employees Checked Out Today
    cursor.execute("SELECT COUNT(*) as completed FROM attendance WHERE attendance_date = ? AND check_out_time IS NOT NULL", (today_date,))
    checked_out_today = cursor.fetchone()['completed']

    # 6. Attendance Percentage
    attendance_percentage = round((present_today / total_employees * 100), 1) if total_employees > 0 else 0.0

    # 7. Today's recent attendance events
    cursor.execute("""
        SELECT a.*, e.full_name, e.department, e.production_block, e.designation, e.photo_path
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.attendance_date = ?
        ORDER BY a.updated_at DESC
        LIMIT 10
    """, (today_date,))
    recent_attendance = [dict(r) for r in cursor.fetchall()]

    # 8. Department-wise breakdown (strictly the 6 designated categories)
    cursor.execute("""
        SELECT e.department, 
               COUNT(e.id) as total,
               COUNT(a.id) as present
        FROM employees e
        LEFT JOIN attendance a ON e.employee_id = a.employee_id AND a.attendance_date = ?
        WHERE e.status = 'active'
        GROUP BY e.department
    """, (today_date,))
    raw_dept_stats = {r['department']: dict(r) for r in cursor.fetchall()}
    dept_stats = []
    for dept_name in VALID_DEPARTMENTS:
        item = raw_dept_stats.get(dept_name, {'department': dept_name, 'total': 0, 'present': 0})
        dept_stats.append(item)

    # 9. Production Block breakdown (PB1, PB2, PB3, Facility Wide)
    cursor.execute("""
        SELECT COALESCE(e.production_block, 'PB1') as production_block, 
               COUNT(e.id) as total,
               COUNT(a.id) as present
        FROM employees e
        LEFT JOIN attendance a ON e.employee_id = a.employee_id AND a.attendance_date = ?
        WHERE e.status = 'active'
        GROUP BY e.production_block
    """, (today_date,))
    raw_block_stats = {r['production_block']: dict(r) for r in cursor.fetchall()}
    
    block_meta = {
        'PB1': {'title': 'Production Block 1 (PB1)', 'desc': 'Tablets & Capsules Formulation'},
        'PB2': {'title': 'Production Block 2 (PB2)', 'desc': 'Liquids & Syrups Processing'},
        'PB3': {'title': 'Production Block 3 (PB3)', 'desc': 'Ointments, Creams & Packaging'},
        'Facility Wide': {'title': 'General Facility Wide', 'desc': 'Security, Operations & Common Staff'}
    }
    block_stats = []
    for blk in VALID_PRODUCTION_BLOCKS:
        item = raw_block_stats.get(blk, {'production_block': blk, 'total': 0, 'present': 0})
        meta = block_meta.get(blk, {'title': blk, 'desc': 'Plant Operations'})
        item['title'] = meta['title']
        item['desc'] = meta['desc']
        item['attendance_rate'] = round((item['present'] / item['total'] * 100), 1) if item['total'] > 0 else 0.0
        block_stats.append(item)

    # 10. Configured Shifts
    cursor.execute("SELECT * FROM shifts ORDER BY is_default DESC, id ASC")
    shifts = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return render_template(
        'index.html',
        active_page='dashboard',
        total_employees=total_employees,
        present_today=present_today,
        absent_today=absent_today,
        currently_checked_in=currently_checked_in,
        checked_out_today=checked_out_today,
        attendance_percentage=attendance_percentage,
        today_date=today_date,
        recent_attendance=recent_attendance,
        dept_stats=dept_stats,
        block_stats=block_stats,
        shifts=shifts
    )


# ==============================================================================
# Employee Registration & Management
# ==============================================================================

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register_employee():
    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == 'POST':
        employee_id = request.form.get('employee_id', '').strip().upper()
        full_name = request.form.get('full_name', '').strip()
        department = request.form.get('department', '').strip()
        production_block = request.form.get('production_block', 'PB1').strip()
        designation = request.form.get('designation', '').strip()
        contact_number = request.form.get('contact_number', '').strip()
        joining_date = request.form.get('joining_date', '').strip()
        shift_id = request.form.get('shift_id', type=int)
        webcam_data = request.form.get('webcam_photo_data', '')

        # 1. Validation
        if not all([employee_id, full_name, department, designation, contact_number, joining_date, shift_id]):
            flash("All mandatory registration fields must be filled.", "danger")
            return redirect(url_for('register_employee'))

        if department not in VALID_DEPARTMENTS:
            flash(f"Department must be one of: {', '.join(VALID_DEPARTMENTS)}", "danger")
            return redirect(url_for('register_employee'))

        if production_block not in VALID_PRODUCTION_BLOCKS:
            production_block = 'PB1'

        cursor.execute("SELECT id FROM employees WHERE employee_id = ?", (employee_id,))
        if cursor.fetchone():
            flash(f"Employee ID '{employee_id}' is already assigned to another employee. Please use a unique ID.", "danger")
            return redirect(url_for('register_employee'))

        # 2. Process Employee Photo
        photo_cv_img = None
        photo_rel_path = None

        if webcam_data and webcam_data.startswith('data:image'):
            # Decoded from base64 webcam capture
            try:
                header, encoded = webcam_data.split(',', 1)
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                photo_cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Save file
                filename = f"{employee_id}.jpg"
                save_dest = os.path.join(PHOTOS_DIR, filename)
                cv2.imwrite(save_dest, photo_cv_img)
                photo_rel_path = f"/static/images/employees/{filename}"
            except Exception as e:
                flash(f"Failed processing webcam photo: {e}", "danger")
                return redirect(url_for('register_employee'))
        elif 'photo' in request.files and request.files['photo'].filename != '':
            # File upload
            file = request.files['photo']
            filename = f"{employee_id}.jpg"
            save_dest = os.path.join(PHOTOS_DIR, filename)
            file.save(save_dest)
            photo_cv_img = cv2.imread(save_dest)
            photo_rel_path = f"/static/images/employees/{filename}"
        else:
            flash("Employee photo is mandatory for biometric face recognition enrollment.", "danger")
            return redirect(url_for('register_employee'))

        # 3. Biometric Face Feature Extraction with YuNet & SFace
        if photo_cv_img is not None:
            face_success, face_msg = face_engine.register_employee_face(employee_id, photo_cv_img)
            if not face_success:
                flash(f"Biometric face enrollment issue: {face_msg}. Employee photo was saved, but facial embeddings may need re-enrolling.", "warning")
            else:
                flash("Face characteristics enrolled successfully into SFace neural database.", "success")

        # 4. Insert Employee Record in SQLite (must exist before child tables reference it)
        cursor.execute("""
            INSERT INTO employees 
            (employee_id, full_name, department, production_block, designation, contact_number, joining_date, shift_id, photo_path, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
        """, (employee_id, full_name, department, production_block, designation, contact_number, joining_date, shift_id, photo_rel_path))

        # 5. Generate Cryptographically Secure QR Token & QR Code Image
        raw_qr_token, token_hash = generate_secure_qr_token(employee_id)
        
        cursor.execute("""
            INSERT INTO employee_qr_tokens (employee_id, token_hash, raw_token_preview, is_active)
            VALUES (?, ?, ?, 1)
        """, (employee_id, token_hash, raw_qr_token[:15] + "..."))

        # Generate styled QR Code PNG
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4
        )
        qr.add_data(raw_qr_token)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#0f172a", back_color="#ffffff")
        qr_file_path = os.path.join(QR_CODES_DIR, f"{employee_id}.png")
        qr_img.save(qr_file_path)

        # 6. Audit Log
        cursor.execute("""
            INSERT INTO attendance_audit_logs (employee_id, action, changed_by, reason, details)
            VALUES (?, 'REGISTER_EMPLOYEE', ?, 'New employee onboarding', ?)
        """, (employee_id, session.get('username', 'admin'), f"Registered {full_name} ({department} - {production_block})"))

        conn.commit()
        conn.close()

        flash(f"Employee {full_name} ({employee_id}) onboarded successfully! QR Code and face characteristics generated.", "success")
        return redirect(url_for('employee_details', employee_id=employee_id))

    # GET: Prepare form data
    cursor.execute("SELECT * FROM shifts ORDER BY is_default DESC, id ASC")
    shifts = [dict(r) for r in cursor.fetchall()]

    # Suggest next ID
    cursor.execute("SELECT COUNT(*) as count FROM employees")
    count = cursor.fetchone()['count'] + 1001
    suggested_id = f"EMP-{count}"

    conn.close()

    return render_template(
        'register_employee.html',
        active_page='register',
        shifts=shifts,
        departments=VALID_DEPARTMENTS,
        production_blocks=VALID_PRODUCTION_BLOCKS,
        suggested_id=suggested_id,
        today_date=date.today().strftime("%Y-%m-%d")
    )

@app.route('/employees')
@login_required
def employees():
    conn = get_db_connection()
    cursor = conn.cursor()

    search = request.args.get('search', '').strip()
    department = request.args.get('department', '').strip()
    production_block = request.args.get('production_block', '').strip()
    status = request.args.get('status', '').strip()
    today_date = date.today().strftime("%Y-%m-%d")

    query = """
        SELECT e.*, s.shift_name, 
               a.check_in_time as today_check_in, 
               a.check_out_time as today_check_out
        FROM employees e
        LEFT JOIN shifts s ON e.shift_id = s.id
        LEFT JOIN attendance a ON e.employee_id = a.employee_id AND a.attendance_date = ?
        WHERE 1=1
    """
    params = [today_date]

    if search:
        query += " AND (e.full_name LIKE ? OR e.employee_id LIKE ? OR e.designation LIKE ?)"
        wildcard = f"%{search}%"
        params.extend([wildcard, wildcard, wildcard])

    if department:
        query += " AND e.department = ?"
        params.append(department)

    if production_block:
        query += " AND e.production_block = ?"
        params.append(production_block)

    if status:
        query += " AND e.status = ?"
        params.append(status)

    query += " ORDER BY e.id DESC"
    cursor.execute(query, params)
    employees_list = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return render_template(
        'employees.html',
        active_page='employees',
        employees=employees_list,
        departments=VALID_DEPARTMENTS,
        production_blocks=VALID_PRODUCTION_BLOCKS
    )

@app.route('/employees/<employee_id>')
@login_required
def employee_details(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e.*, s.shift_name 
        FROM employees e
        LEFT JOIN shifts s ON e.shift_id = s.id
        WHERE e.employee_id = ?
    """, (employee_id,))
    emp = cursor.fetchone()

    if not emp:
        conn.close()
        flash(f"Employee '{employee_id}' not found.", "danger")
        return redirect(url_for('employees'))

    # QR Code image check
    qr_filename = f"{employee_id}.png"
    qr_image_path = f"/qr_codes/{qr_filename}" if os.path.exists(os.path.join(QR_CODES_DIR, qr_filename)) else None

    # Attendance logs for this employee
    cursor.execute("""
        SELECT * FROM attendance 
        WHERE employee_id = ? 
        ORDER BY attendance_date DESC 
        LIMIT 30
    """, (employee_id,))
    attendance_records = [dict(r) for r in cursor.fetchall()]

    # Statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_present,
            SUM(is_late) as total_late,
            SUM(total_working_minutes) as total_minutes
        FROM attendance
        WHERE employee_id = ?
    """, (employee_id,))
    stats = cursor.fetchone()

    # Shifts for edit modal
    cursor.execute("SELECT * FROM shifts ORDER BY id ASC")
    shifts = [dict(r) for r in cursor.fetchall()]

    conn.close()

    has_face_model = employee_id in face_engine.embeddings

    return render_template(
        'employee_details.html',
        active_page='employees',
        employee=dict(emp),
        qr_image_path=qr_image_path,
        attendance_records=attendance_records,
        total_present=stats['total_present'] or 0,
        total_late=stats['total_late'] or 0,
        total_minutes=stats['total_minutes'] or 0,
        has_face_model=has_face_model,
        departments=VALID_DEPARTMENTS,
        production_blocks=VALID_PRODUCTION_BLOCKS,
        shifts=shifts
    )

@app.route('/employees/<employee_id>/edit', methods=['POST'])
@login_required
def edit_employee(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    full_name = request.form.get('full_name', '').strip()
    department = request.form.get('department', '').strip()
    production_block = request.form.get('production_block', 'PB1').strip()
    designation = request.form.get('designation', '').strip()
    contact_number = request.form.get('contact_number', '').strip()
    shift_id = request.form.get('shift_id', type=int)

    if not all([full_name, department, designation, contact_number]):
        flash("All profile fields are required.", "danger")
        return redirect(url_for('employee_details', employee_id=employee_id))

    if department not in VALID_DEPARTMENTS:
        department = 'Workers'
    if production_block not in VALID_PRODUCTION_BLOCKS:
        production_block = 'PB1'

    # Optional photo update
    photo_rel_path = None
    if 'photo' in request.files and request.files['photo'].filename != '':
        file = request.files['photo']
        filename = f"{employee_id}.jpg"
        save_dest = os.path.join(PHOTOS_DIR, filename)
        file.save(save_dest)
        photo_rel_path = f"/static/images/employees/{filename}"
        
        # Update facial embedding
        cv_img = cv2.imread(save_dest)
        if cv_img is not None:
            face_engine.register_employee_face(employee_id, cv_img)

    if photo_rel_path:
        cursor.execute("""
            UPDATE employees 
            SET full_name = ?, department = ?, production_block = ?, designation = ?, contact_number = ?, shift_id = ?, photo_path = ?, updated_at = CURRENT_TIMESTAMP
            WHERE employee_id = ?
        """, (full_name, department, production_block, designation, contact_number, shift_id, photo_rel_path, employee_id))
    else:
        cursor.execute("""
            UPDATE employees 
            SET full_name = ?, department = ?, production_block = ?, designation = ?, contact_number = ?, shift_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE employee_id = ?
        """, (full_name, department, production_block, designation, contact_number, shift_id, employee_id))

    cursor.execute("""
        INSERT INTO attendance_audit_logs (employee_id, action, changed_by, reason)
        VALUES (?, 'UPDATE_PROFILE', ?, 'Updated employee profile details')
    """, (employee_id, session.get('username', 'admin')))

    conn.commit()
    conn.close()

    flash(f"Profile updated successfully for {full_name}.", "success")
    return redirect(url_for('employee_details', employee_id=employee_id))

@app.route('/employees/<employee_id>/deactivate', methods=['POST'])
@login_required
def deactivate_employee(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE employees SET status = 'inactive', updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?", (employee_id,))
    cursor.execute("""
        INSERT INTO attendance_audit_logs (employee_id, action, changed_by, reason)
        VALUES (?, 'DEACTIVATE_EMPLOYEE', ?, 'Employee deactivated by administrator')
    """, (employee_id, session.get('username', 'admin')))

    conn.commit()
    conn.close()

    flash(f"Employee {employee_id} has been deactivated. Historical records remain preserved.", "warning")
    return redirect(request.referrer or url_for('employees'))

@app.route('/employees/<employee_id>/reactivate', methods=['POST'])
@login_required
def reactivate_employee(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE employees SET status = 'active', updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?", (employee_id,))
    cursor.execute("""
        INSERT INTO attendance_audit_logs (employee_id, action, changed_by, reason)
        VALUES (?, 'REACTIVATE_EMPLOYEE', ?, 'Employee reactivated by administrator')
    """, (employee_id, session.get('username', 'admin')))

    conn.commit()
    conn.close()

    flash(f"Employee {employee_id} reactivated successfully.", "success")
    return redirect(request.referrer or url_for('employees'))

@app.route('/employees/<employee_id>/delete', methods=['POST'])
@login_required
def delete_employee(employee_id):
    # Verify CSRF token
    csrf_token = request.form.get('csrf_token')
    if not csrf_token or csrf_token != session.get('csrf_token'):
        flash("Security token validation failed. Please try again.", "danger")
        return redirect(url_for('employees'))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
    emp = cursor.fetchone()

    if not emp:
        conn.close()
        flash(f"Employee {employee_id} not found.", "danger")
        return redirect(url_for('employees'))

    full_name = emp['full_name']

    # 1. Delete face embedding via FaceEngine
    try:
        face_engine.delete_employee_face(employee_id)
    except Exception as e:
        print(f"[delete_employee] Error removing face embedding for {employee_id}: {e}")

    # 2. Delete photo file from PHOTOS_DIR
    for ext in ['.jpg', '.jpeg', '.png']:
        photo_file = os.path.join(PHOTOS_DIR, f"{employee_id}{ext}")
        if os.path.exists(photo_file):
            try:
                os.remove(photo_file)
            except Exception as e:
                print(f"[delete_employee] Error deleting photo {photo_file}: {e}")

    # 3. Delete QR code image file
    qr_file = os.path.join(QR_CODES_DIR, f"{employee_id}.png")
    if os.path.exists(qr_file):
        try:
            os.remove(qr_file)
        except Exception as e:
            print(f"[delete_employee] Error deleting QR file {qr_file}: {e}")

    # 4. Remove all database records
    cursor.execute("DELETE FROM break_logs WHERE employee_id = ?", (employee_id,))
    cursor.execute("DELETE FROM attendance WHERE employee_id = ?", (employee_id,))
    cursor.execute("DELETE FROM employee_qr_tokens WHERE employee_id = ?", (employee_id,))
    cursor.execute("DELETE FROM attendance_audit_logs WHERE employee_id = ?", (employee_id,))
    cursor.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))

    # Log deletion in audit logs
    cursor.execute("""
        INSERT INTO attendance_audit_logs (employee_id, action, changed_by, reason)
        VALUES (?, 'DELETE_EMPLOYEE', ?, ?)
    """, ('SYSTEM', session.get('username', 'admin'), f"Permanently deleted employee {full_name} ({employee_id})"))

    conn.commit()
    conn.close()

    flash(f"Employee {full_name} ({employee_id}) and all associated records have been permanently deleted.", "success")
    return redirect(url_for('employees'))

@app.route('/employees/<employee_id>/regenerate_qr', methods=['POST'])
@login_required
def regenerate_qr(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Revoke old tokens
    cursor.execute("UPDATE employee_qr_tokens SET is_active = 0, revoked_at = CURRENT_TIMESTAMP WHERE employee_id = ?", (employee_id,))

    # Generate new token
    raw_token, token_hash = generate_secure_qr_token(employee_id)
    cursor.execute("""
        INSERT INTO employee_qr_tokens (employee_id, token_hash, raw_token_preview, is_active)
        VALUES (?, ?, ?, 1)
    """, (employee_id, token_hash, raw_token[:15] + "..."))

    # Generate new QR PNG
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(raw_token)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="#0f172a", back_color="#ffffff")
    qr_img.save(os.path.join(QR_CODES_DIR, f"{employee_id}.png"))

    cursor.execute("""
        INSERT INTO attendance_audit_logs (employee_id, action, changed_by, reason)
        VALUES (?, 'REGENERATE_QR', ?, 'Revoked old QR token and generated new token')
    """, (employee_id, session.get('username', 'admin')))

    conn.commit()
    conn.close()

    flash(f"New QR Code generated and previous tokens revoked for {employee_id}.", "success")
    return redirect(url_for('employee_details', employee_id=employee_id))

@app.route('/qr_codes/<path:filename>')
@login_required
def serve_qr_code(filename):
    return send_from_directory(QR_CODES_DIR, filename)


# ==============================================================================
# Face Recognition Attendance Kiosk & API
# ==============================================================================

@app.route('/face_attendance')
@login_required
def face_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()

    today_date = date.today().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT a.*, e.full_name, e.department, e.production_block
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.attendance_date = ? AND a.attendance_method = 'Face Recognition'
        ORDER BY a.updated_at DESC
        LIMIT 10
    """, (today_date,))
    session_recent = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template(
        'face_attendance.html',
        active_page='face_attendance',
        enrolled_faces_count=len(face_engine.embeddings),
        session_recent=session_recent
    )

@app.route('/api/recognize_face', methods=['POST'])
def api_recognize_face():
    """
    Receives base64 JPEG from webcam, detects face with YuNet, extracts features with SFace,
    matches against registered employee embeddings, and records attendance.
    """
    data = request.get_json() or {}
    base64_img = data.get('image', '')

    if not base64_img or ',' not in base64_img:
        return jsonify({"success": False, "error": "No valid image data supplied."}), 400

    try:
        header, encoded = base64_img.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Unable to decode camera frame."}), 400

        # Perform YuNet detection & SFace matching
        recognition_res = face_engine.recognize_from_bgr_frame(frame)

        if not recognition_res['success']:
            return jsonify(recognition_res)

        if not recognition_res['is_match'] or not recognition_res['matched_employee_id']:
            return jsonify({
                "success": True,
                "is_match": False,
                "confidence_percent": recognition_res['confidence_percent'],
                "cosine_score": recognition_res['cosine_score'],
                "faces_detected": recognition_res['faces_detected']
            })

        # Match succeeded! Record attendance
        matched_id = recognition_res['matched_employee_id']
        event_type = data.get('event_type', 'auto')
        att_res = process_attendance_event(matched_id, 'Face Recognition', event_type=event_type)

        if not att_res['success']:
            return jsonify({
                "success": False,
                "is_match": True,
                "is_completed": att_res.get('is_completed', False),
                "error": att_res['error'],
                "confidence_percent": recognition_res['confidence_percent'],
                "cosine_score": recognition_res['cosine_score']
            })

        emp_info = att_res['employee']
        return jsonify({
            "success": True,
            "is_match": True,
            "confidence_percent": recognition_res['confidence_percent'],
            "cosine_score": recognition_res['cosine_score'],
            "employee": {
                "employee_id": emp_info['employee_id'],
                "full_name": emp_info['full_name'],
                "department": emp_info['department'],
                "production_block": emp_info.get('production_block', 'PB1'),
                "designation": emp_info['designation'],
                "photo_url": emp_info['photo_path']
            },
            "attendance": att_res['attendance'],
            "message": att_res['message']
        })

    except Exception as e:
        print(f"Error in api_recognize_face: {e}")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


# ==============================================================================
# QR Code Attendance Kiosk & API
# ==============================================================================

@app.route('/scan_qr')
@login_required
def scan_qr():
    conn = get_db_connection()
    cursor = conn.cursor()

    today_date = date.today().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT a.*, e.full_name, e.department, e.production_block
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.attendance_date = ? AND a.attendance_method = 'QR Code'
        ORDER BY a.updated_at DESC
        LIMIT 10
    """, (today_date,))
    session_recent = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template(
        'scan_qr.html',
        active_page='scan_qr',
        session_recent=session_recent
    )

@app.route('/api/scan_qr', methods=['POST'])
def api_scan_qr():
    """
    Validates scanned QR raw token by hashing with SHA-256 and querying employee_qr_tokens.
    """
    data = request.get_json() or {}
    raw_token = data.get('token', '').strip()

    if not raw_token:
        return jsonify({"success": False, "error": "No QR token provided."}), 400

    token_hash = hash_qr_token(raw_token)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT employee_id 
        FROM employee_qr_tokens 
        WHERE token_hash = ? AND is_active = 1
    """, (token_hash,))
    token_record = cursor.fetchone()
    conn.close()

    if not token_record:
        return jsonify({"success": False, "error": "Invalid, unverified, or revoked QR badge."}), 404

    employee_id = token_record['employee_id']
    event_type = data.get('event_type', 'auto')
    att_res = process_attendance_event(employee_id, 'QR Code', event_type=event_type)

    if not att_res['success']:
        return jsonify({
            "success": False,
            "is_completed": att_res.get('is_completed', False),
            "error": att_res['error']
        })

    emp_info = att_res['employee']
    return jsonify({
        "success": True,
        "employee": {
            "employee_id": emp_info['employee_id'],
            "full_name": emp_info['full_name'],
            "department": emp_info['department'],
            "production_block": emp_info.get('production_block', 'PB1'),
            "designation": emp_info['designation'],
            "photo_url": emp_info['photo_path']
        },
        "attendance": att_res['attendance'],
        "message": att_res['message']
    })

@app.route('/api/scan_qr_image', methods=['POST'])
def api_scan_qr_image():
    """Fallback decoder: decodes QR code from uploaded image using OpenCV QRCodeDetector."""
    data = request.get_json() or {}
    base64_img = data.get('image', '')
    event_type = data.get('event_type', 'auto')

    if not base64_img or ',' not in base64_img:
        return jsonify({"success": False, "error": "Invalid image payload."}), 400

    try:
        header, encoded = base64_img.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detector = cv2.QRCodeDetector()
        val, points, qrcode_mat = detector.detectAndDecode(img)

        if not val:
            return jsonify({"success": False, "error": "No QR code could be decoded from image."}), 400

        # Validate decoded token
        token_hash = hash_qr_token(val)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT employee_id FROM employee_qr_tokens WHERE token_hash = ? AND is_active = 1", (token_hash,))
        token_record = cursor.fetchone()
        conn.close()

        if not token_record:
            return jsonify({"success": False, "error": "Decoded QR token is invalid or revoked."}), 404

        att_res = process_attendance_event(token_record['employee_id'], 'QR Code', event_type=event_type)
        if not att_res['success']:
            return jsonify({
                "success": False,
                "is_completed": att_res.get('is_completed', False),
                "error": att_res['error']
            })

        emp_info = att_res['employee']
        return jsonify({
            "success": True,
            "employee": {
                "employee_id": emp_info['employee_id'],
                "full_name": emp_info['full_name'],
                "department": emp_info['department'],
                "designation": emp_info['designation'],
                "photo_url": emp_info['photo_path']
            },
            "attendance": att_res['attendance'],
            "message": att_res['message']
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==============================================================================
# Attendance History & Audit Corrections
# ==============================================================================

@app.route('/attendance')
@login_required
def attendance_history():
    conn = get_db_connection()
    cursor = conn.cursor()

    page = request.args.get('page', 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page

    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    search = request.args.get('search', '').strip()
    department = request.args.get('department', '').strip()
    production_block = request.args.get('production_block', '').strip()
    status = request.args.get('status', '').strip()
    method = request.args.get('method', '').strip()

    query = """
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE 1=1
    """
    params = []

    if date_from:
        query += " AND a.attendance_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND a.attendance_date <= ?"
        params.append(date_to)
    if search:
        query += " AND (e.full_name LIKE ? OR e.employee_id LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    if department:
        query += " AND e.department = ?"
        params.append(department)
    if production_block:
        query += " AND e.production_block = ?"
        params.append(production_block)
    if status:
        query += " AND a.status = ?"
        params.append(status)
    if method:
        query += " AND a.attendance_method = ?"
        params.append(method)

    # Count total records
    count_cursor = conn.cursor()
    count_cursor.execute("SELECT COUNT(*) as cnt " + query, params)
    total_records = count_cursor.fetchone()['cnt']
    total_pages = max((total_records + per_page - 1) // per_page, 1)

    # Fetch rows
    select_query = """
        SELECT a.*, e.full_name, e.department, e.production_block
    """ + query + " ORDER BY a.attendance_date DESC, a.check_in_time DESC LIMIT ? OFFSET ?"
    
    fetch_params = list(params) + [per_page, offset]
    cursor.execute(select_query, fetch_params)
    records = [dict(r) for r in cursor.fetchall()]

    conn.close()

    # Pass request arguments for clean pagination URLs
    args_dict = request.args.to_dict()
    args_dict.pop('page', None)

    return render_template(
        'attendance.html',
        active_page='attendance',
        records=records,
        departments=VALID_DEPARTMENTS,
        production_blocks=VALID_PRODUCTION_BLOCKS,
        total_records=total_records,
        current_page=page,
        total_pages=total_pages,
        request_args_without_page=args_dict
    )

@app.route('/attendance/correct', methods=['POST'])
@login_required
def correct_attendance_record():
    """Manual administrator override / correction with mandatory audit trail log."""
    record_id = request.form.get('record_id', type=int)
    check_in_time = request.form.get('check_in_time', '').strip()
    check_out_time = request.form.get('check_out_time', '').strip()
    status = request.form.get('status', '').strip()
    reason = request.form.get('reason', '').strip()

    if not record_id or not reason:
        flash("Record ID and Audit Reason are mandatory for manual correction.", "danger")
        return redirect(url_for('attendance_history'))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance WHERE id = ?", (record_id,))
    old_record = cursor.fetchone()

    if not old_record:
        conn.close()
        flash("Record not found.", "danger")
        return redirect(url_for('attendance_history'))

    # Recalculate duration if both times provided
    total_mins = 0
    if check_in_time and check_out_time:
        total_mins = calculate_session_duration(check_in_time, check_out_time)

    cursor.execute("""
        UPDATE attendance
        SET check_in_time = ?,
            check_out_time = ?,
            status = ?,
            total_working_minutes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (check_in_time or None, check_out_time or None, status, total_mins, record_id))

    # Save to audit trail
    details = f"Old: In={old_record['check_in_time']}, Out={old_record['check_out_time']}, Status={old_record['status']} | New: In={check_in_time}, Out={check_out_time}, Status={status}"
    cursor.execute("""
        INSERT INTO attendance_audit_logs (employee_id, attendance_id, action, changed_by, reason, details)
        VALUES (?, ?, 'MANUAL_OVERRIDE', ?, ?, ?)
    """, (old_record['employee_id'], record_id, session.get('username', 'admin'), reason, details))

    conn.commit()
    conn.close()

    flash("Attendance record updated and audit trail logged successfully.", "success")
    return redirect(request.referrer or url_for('attendance_history'))


# ==============================================================================
# Reports & Analytics
# ==============================================================================

@app.route('/reports')
@login_required
def reports():
    conn = get_db_connection()
    cursor = conn.cursor()

    selected_report = request.args.get('report_type', 'daily')
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    selected_dept = request.args.get('department', '').strip()
    selected_block = request.args.get('production_block', '').strip()

    today_str = date.today().strftime("%Y-%m-%d")

    # Default date ranges based on report
    if not date_from or not date_to:
        if selected_report == 'daily':
            date_from = date_to = today_str
        elif selected_report == 'weekly':
            date_from = (date.today() - timedelta(days=6)).strftime("%Y-%m-%d")
            date_to = today_str
        else:
            date_from = (date.today() - timedelta(days=29)).strftime("%Y-%m-%d")
            date_to = today_str

    report_title = "Report"
    report_headers = []
    report_rows = []

    # 1. Daily, Weekly, Monthly Detailed Report
    if selected_report in ['daily', 'weekly', 'monthly']:
        report_title = f"{selected_report.capitalize()} Detailed Attendance Log"
        report_headers = ["Date", "Employee ID", "Name", "Department", "Block", "Check-In", "Tea Break", "Lunch Break", "Check-Out", "Net Productive Hours", "Status", "Method"]
        
        query = """
            SELECT a.attendance_date, a.employee_id, e.full_name, e.department, e.production_block,
                   a.check_in_time, a.tea_break_minutes, a.lunch_break_minutes, a.total_break_minutes,
                   a.check_out_time, a.total_working_minutes, a.effective_working_minutes, a.status, a.attendance_method
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
            WHERE a.attendance_date BETWEEN ? AND ?
        """
        params = [date_from, date_to]
        if selected_dept:
            query += " AND e.department = ?"
            params.append(selected_dept)
        if selected_block:
            query += " AND e.production_block = ?"
            params.append(selected_block)
        query += " ORDER BY a.attendance_date DESC, a.check_in_time ASC"
        
        cursor.execute(query, params)
        for r in cursor.fetchall():
            tea_m = f"{r['tea_break_minutes']}m" if r['tea_break_minutes'] and r['tea_break_minutes'] > 0 else "-"
            lunch_m = f"{r['lunch_break_minutes']}m" if r['lunch_break_minutes'] and r['lunch_break_minutes'] > 0 else "-"
            net_mins = r['effective_working_minutes'] or r['total_working_minutes'] or 0
            hrs_str = f"{net_mins // 60}h {net_mins % 60}m" if net_mins > 0 else "-"
            report_rows.append([
                r['attendance_date'], r['employee_id'], r['full_name'], r['department'], r['production_block'] or 'PB1',
                r['check_in_time'] or "-", tea_m, lunch_m, r['check_out_time'] or "-", hrs_str, r['status'], r['attendance_method']
            ])

    # 2. Employee-Wise Summary
    elif selected_report == 'employee_wise':
        report_title = "Employee-Wise Cumulative Attendance Summary"
        report_headers = ["Employee ID", "Full Name", "Department", "Block", "Designation", "Days Present", "Late Count", "Total Hours", "Avg Hours/Day"]
        
        query = """
            SELECT e.employee_id, e.full_name, e.department, e.production_block, e.designation,
                   COUNT(a.id) as days_present,
                   SUM(CASE WHEN a.is_late = 1 THEN 1 ELSE 0 END) as late_count,
                   SUM(COALESCE(a.effective_working_minutes, a.total_working_minutes, 0)) as total_mins
            FROM employees e
            LEFT JOIN attendance a ON e.employee_id = a.employee_id AND a.attendance_date BETWEEN ? AND ?
            WHERE e.status = 'active'
        """
        params = [date_from, date_to]
        if selected_dept:
            query += " AND e.department = ?"
            params.append(selected_dept)
        if selected_block:
            query += " AND e.production_block = ?"
            params.append(selected_block)
        query += " GROUP BY e.employee_id ORDER BY days_present DESC"

        cursor.execute(query, params)
        for r in cursor.fetchall():
            tot_mins = r['total_mins'] or 0
            tot_hrs = round(tot_mins / 60.0, 1)
            days = r['days_present'] or 0
            avg_hrs = round(tot_hrs / days, 1) if days > 0 else 0.0
            report_rows.append([
                r['employee_id'], r['full_name'], r['department'], r['production_block'] or 'PB1', r['designation'],
                days, r['late_count'] or 0, f"{tot_hrs} hrs", f"{avg_hrs} hrs"
            ])

    # 3. Department-Wise Summary
    elif selected_report == 'department_wise':
        report_title = "Department-Wise Attendance Performance"
        report_headers = ["Department", "Active Staff", "Logged Sessions", "Late Incidents", "Attendance Rate", "Total Work Hours"]
        
        cursor.execute("""
            SELECT e.department,
                   COUNT(DISTINCT e.id) as staff_count,
                   COUNT(a.id) as sessions,
                   SUM(CASE WHEN a.is_late = 1 THEN 1 ELSE 0 END) as late_count,
                   SUM(COALESCE(a.effective_working_minutes, a.total_working_minutes, 0)) as total_mins
            FROM employees e
            LEFT JOIN attendance a ON e.employee_id = a.employee_id AND a.attendance_date BETWEEN ? AND ?
            WHERE e.status = 'active'
            GROUP BY e.department
            ORDER BY staff_count DESC
        """, (date_from, date_to))
        
        for r in cursor.fetchall():
            staff = r['staff_count'] or 0
            sess = r['sessions'] or 0
            tot_mins = r['total_mins'] or 0
            tot_hrs = round(tot_mins / 60.0, 1)
            days_span = max((datetime.strptime(date_to, "%Y-%m-%d") - datetime.strptime(date_from, "%Y-%m-%d")).days + 1, 1)
            expected = staff * days_span
            rate = round((sess / expected * 100), 1) if expected > 0 else 0.0
            report_rows.append([
                r['department'], staff, sess, r['late_count'] or 0, f"{rate}%", f"{tot_hrs} hrs"
            ])

    # 4. Production Block-Wise Summary (PB1, PB2, PB3)
    elif selected_report == 'block_wise':
        report_title = "Production Blocks (PB1 / PB2 / PB3) Workforce Allocation & Performance"
        report_headers = ["Block ID", "Facility / Operation Unit", "Active Headcount", "Logged Sessions", "Late Incidents", "Attendance Rate", "Cumulative Net Hours"]
        
        cursor.execute("""
            SELECT COALESCE(e.production_block, 'PB1') as production_block,
                   COUNT(DISTINCT e.id) as staff_count,
                   COUNT(a.id) as sessions,
                   SUM(CASE WHEN a.is_late = 1 THEN 1 ELSE 0 END) as late_count,
                   SUM(COALESCE(a.effective_working_minutes, a.total_working_minutes, 0)) as total_mins
            FROM employees e
            LEFT JOIN attendance a ON e.employee_id = a.employee_id AND a.attendance_date BETWEEN ? AND ?
            WHERE e.status = 'active'
            GROUP BY e.production_block
            ORDER BY production_block ASC
        """, (date_from, date_to))
        
        block_names = {
            'PB1': 'Tablet & Capsule Formulation (PB1)',
            'PB2': 'Liquid & Syrup Processing (PB2)',
            'PB3': 'Ointment, Cream & Packaging (PB3)',
            'Facility Wide': 'General Operations, Security & Facility'
        }
        days_span = max((datetime.strptime(date_to, "%Y-%m-%d") - datetime.strptime(date_from, "%Y-%m-%d")).days + 1, 1)
        for r in cursor.fetchall():
            blk = r['production_block']
            staff = r['staff_count'] or 0
            sess = r['sessions'] or 0
            tot_hrs = round((r['total_mins'] or 0) / 60.0, 1)
            expected = staff * days_span
            rate = round((sess / expected * 100), 1) if expected > 0 else 0.0
            report_rows.append([
                blk, block_names.get(blk, blk), staff, sess, r['late_count'] or 0, f"{rate}%", f"{tot_hrs} hrs"
            ])

    # 5. Late Arrivals Report
    elif selected_report == 'late_arrivals':
        report_title = "Late Arrivals & Punctuality Audit Report"
        report_headers = ["Date", "Employee ID", "Full Name", "Department", "Block", "Check-In Time", "Shift Start", "Grace Mins", "Status"]
        
        query = """
            SELECT a.attendance_date, a.employee_id, e.full_name, e.department, e.production_block,
                   a.check_in_time, s.start_time, s.grace_period_minutes, a.status
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
            LEFT JOIN shifts s ON e.shift_id = s.id
            WHERE a.is_late = 1 AND a.attendance_date BETWEEN ? AND ?
        """
        params = [date_from, date_to]
        if selected_dept:
            query += " AND e.department = ?"
            params.append(selected_dept)
        if selected_block:
            query += " AND e.production_block = ?"
            params.append(selected_block)
        query += " ORDER BY a.attendance_date DESC, a.check_in_time ASC"

        cursor.execute(query, params)
        for r in cursor.fetchall():
            report_rows.append([
                r['attendance_date'], r['employee_id'], r['full_name'], r['department'], r['production_block'] or 'PB1',
                r['check_in_time'], r['start_time'] or "09:00", f"{r['grace_period_minutes']}m", "Late"
            ])

    # 6. Absence Report
    elif selected_report == 'absence':
        report_title = f"Absence Report for {date_to}"
        report_headers = ["Employee ID", "Full Name", "Department", "Block", "Designation", "Contact", "Shift", "Status"]
        
        query = """
            SELECT e.employee_id, e.full_name, e.department, e.production_block, e.designation, e.contact_number, s.shift_name
            FROM employees e
            LEFT JOIN shifts s ON e.shift_id = s.id
            WHERE e.status = 'active'
              AND e.employee_id NOT IN (
                  SELECT employee_id FROM attendance WHERE attendance_date = ?
              )
        """
        params = [date_to]
        if selected_dept:
            query += " AND e.department = ?"
            params.append(selected_dept)
        if selected_block:
            query += " AND e.production_block = ?"
            params.append(selected_block)
        query += " ORDER BY e.department, e.full_name"

        cursor.execute(query, params)
        for r in cursor.fetchall():
            report_rows.append([
                r['employee_id'], r['full_name'], r['department'], r['production_block'] or 'PB1', r['designation'],
                r['contact_number'], r['shift_name'] or "General", "Absent"
            ])

    # KPI summary calculations
    cursor.execute("""
        SELECT 
            COUNT(*) as total_sessions,
            SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END) as late_count,
            SUM(COALESCE(effective_working_minutes, total_working_minutes, 0)) as total_mins
        FROM attendance
        WHERE attendance_date BETWEEN ? AND ?
    """, (date_from, date_to))
    kpi = cursor.fetchone()
    
    tot_sess = kpi['total_sessions'] or 0
    late_cnt = kpi['late_count'] or 0
    tot_hrs = round((kpi['total_mins'] or 0) / 60.0, 1)
    late_rate = round((late_cnt / tot_sess * 100), 1) if tot_sess > 0 else 0.0
    avg_hrs = round(tot_hrs / tot_sess, 1) if tot_sess > 0 else 0.0

    report_summary = {
        "total_sessions": tot_sess,
        "late_count": late_cnt,
        "late_rate": late_rate,
        "total_hours": tot_hrs,
        "avg_hours_per_session": avg_hrs
    }

    conn.close()

    return render_template(
        'reports.html',
        active_page='reports',
        selected_report=selected_report,
        date_from=date_from,
        date_to=date_to,
        selected_dept=selected_dept,
        selected_block=selected_block,
        report_title=report_title,
        report_headers=report_headers,
        report_rows=report_rows,
        report_summary=report_summary,
        departments=VALID_DEPARTMENTS,
        production_blocks=VALID_PRODUCTION_BLOCKS
    )

@app.route('/reports/export')
@login_required
def export_report_csv():
    """Exports CSV of filtered attendance or reports."""
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    department = request.args.get('department', '').strip()
    production_block = request.args.get('production_block', '').strip()
    status = request.args.get('status', '').strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT a.attendance_date, a.employee_id, e.full_name, e.department, e.production_block, e.designation,
               a.check_in_time, a.tea_break_minutes, a.lunch_break_minutes, a.total_break_minutes,
               a.check_out_time, a.total_working_minutes, a.effective_working_minutes, a.status, a.is_late, a.attendance_method
        FROM attendance a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE 1=1
    """
    params = []
    if date_from:
        query += " AND a.attendance_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND a.attendance_date <= ?"
        params.append(date_to)
    if department:
        query += " AND e.department = ?"
        params.append(department)
    if production_block:
        query += " AND e.production_block = ?"
        params.append(production_block)
    if status:
        query += " AND a.status = ?"
        params.append(status)

    query += " ORDER BY a.attendance_date DESC, a.check_in_time ASC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Generate CSV stream
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Attendance Date", "Employee ID", "Employee Name", "Department", "Production Block", "Designation",
        "Check-In Time", "Tea Break (Mins)", "Lunch Break (Mins)", "Total Breaks (Mins)",
        "Check-Out Time", "Gross Hours", "Net Productive Hours", "Status", "Late Arrival", "Method"
    ])

    for r in rows:
        gross_mins = r['total_working_minutes'] or 0
        net_mins = r['effective_working_minutes'] or gross_mins
        gross_hrs = round(gross_mins / 60.0, 2)
        net_hrs = round(net_mins / 60.0, 2)
        writer.writerow([
            r['attendance_date'], r['employee_id'], r['full_name'], r['department'], r['production_block'] or 'PB1', r['designation'],
            r['check_in_time'] or "", r['tea_break_minutes'] or 0, r['lunch_break_minutes'] or 0, r['total_break_minutes'] or 0,
            r['check_out_time'] or "", gross_hrs, net_hrs, r['status'],
            "Yes" if r['is_late'] else "No", r['attendance_method']
        ])

    csv_data = output.getvalue()
    filename = f"attendance_report_{date.today().strftime('%Y%m%d')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ==============================================================================
# Settings & Shift Configuration
# ==============================================================================

@app.route('/settings')
@login_required
def settings():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT key, value FROM company_settings")
    settings_dict = {r['key']: r['value'] for r in cursor.fetchall()}

    cursor.execute("SELECT * FROM shifts ORDER BY id ASC")
    shifts = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return render_template(
        'settings.html',
        active_page='settings',
        settings_dict=settings_dict,
        shifts=shifts
    )

@app.route('/settings/company', methods=['POST'])
@login_required
def update_company_settings():
    company_name = request.form.get('company_name', '').strip()
    full_day_hours = request.form.get('work_hours_full_day', '8').strip()
    half_day_hours = request.form.get('work_hours_half_day', '4').strip()
    timezone = request.form.get('timezone', 'Asia/Kolkata').strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    updates = [
        ('company_name', company_name),
        ('work_hours_full_day', full_day_hours),
        ('work_hours_half_day', half_day_hours),
        ('timezone', timezone)
    ]
    for k, v in updates:
        cursor.execute("INSERT OR REPLACE INTO company_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (k, v))

    conn.commit()
    conn.close()

    flash("Company configuration updated successfully.", "success")
    return redirect(url_for('settings'))

@app.route('/settings/shifts/add', methods=['POST'])
@login_required
def add_shift():
    shift_name = request.form.get('shift_name', '').strip()
    start_time = request.form.get('start_time', '').strip()
    end_time = request.form.get('end_time', '').strip()
    grace_period = request.form.get('grace_period_minutes', 15, type=int)

    if not all([shift_name, start_time, end_time]):
        flash("All shift configuration fields are required.", "danger")
        return redirect(url_for('settings'))

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO shifts (shift_name, start_time, end_time, grace_period_minutes, is_default)
            VALUES (?, ?, ?, ?, 0)
        """, (shift_name, start_time, end_time, grace_period))
        conn.commit()
        flash(f"Shift '{shift_name}' created successfully.", "success")
    except Exception as e:
        flash(f"Error creating shift: {e}", "danger")
    finally:
        conn.close()

    return redirect(url_for('settings'))

@app.route('/settings/password', methods=['POST'])
@login_required
def change_password():
    current_pwd = request.form.get('current_password', '')
    new_pwd = request.form.get('new_password', '')
    confirm_pwd = request.form.get('confirm_password', '')

    if new_pwd != confirm_pwd:
        flash("New passwords do not match.", "danger")
        return redirect(url_for('settings'))

    if len(new_pwd) < 6:
        flash("Password must be at least 6 characters.", "danger")
        return redirect(url_for('settings'))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT password_hash FROM admin_users WHERE id = ?", (session['user_id'],))
    user = cursor.fetchone()

    if not user or not check_password_hash(user['password_hash'], current_pwd):
        conn.close()
        flash("Current password is incorrect.", "danger")
        return redirect(url_for('settings'))

    new_hash = generate_password_hash(new_pwd)
    cursor.execute("UPDATE admin_users SET password_hash = ? WHERE id = ?", (new_hash, session['user_id']))
    conn.commit()
    conn.close()

    flash("Administrator password updated successfully.", "success")
    return redirect(url_for('settings'))


# ==============================================================================
# Application Startup
# ==============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" Himalaya Wellness Company - Smart Attendance System")
    print(f" Web Portal running at: http://127.0.0.1:{port}")
    print(f" Administrator Account: darshan")
    print(f" Face Recognition Model: YuNet + SFace (ONNX)")
    print(f" QR Code Engine:        SHA-256 Revocable Tokens")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=True)
