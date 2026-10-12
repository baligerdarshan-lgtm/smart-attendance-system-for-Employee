"""
Database module for Smart Attendance Management System
Manages SQLite schema, migrations, connection handling, and seed data.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Support serverless read-only filesystems (e.g. Vercel, AWS Lambda)
IS_SERVERLESS = bool(os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME'))
if IS_SERVERLESS:
    DB_PATH = os.path.join('/tmp', 'attendance.db')
    # If seeded db exists in repo, copy it to /tmp on first run
    repo_db = os.path.join(BASE_DIR, 'database', 'attendance.db')
    if os.path.exists(repo_db) and not os.path.exists(DB_PATH):
        try:
            import shutil
            shutil.copyfile(repo_db, DB_PATH)
        except Exception:
            pass
else:
    DB_PATH = os.path.join(BASE_DIR, 'database', 'attendance.db')

def get_db_connection():
    """Returns a SQLite connection with Row factory enabled and foreign keys enforced."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def hash_qr_token(raw_token: str) -> str:
    """Computes SHA-256 hash of a raw QR token string."""
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()

def generate_secure_qr_token(employee_id: str) -> tuple[str, str]:
    """
    Generates a cryptographically secure token for an employee.
    Returns (raw_token, token_hash).
    The raw token format is: EMPQR_<employee_id>_<random_hex>
    """
    random_part = secrets.token_hex(16)
    raw_token = f"EMPQR_{employee_id}_{random_part}"
    token_hash = hash_qr_token(raw_token)
    return raw_token, token_hash

VALID_DEPARTMENTS = ['Managers', 'Supervisors', 'Employees', 'Workers', 'Security', 'Others']
VALID_PRODUCTION_BLOCKS = ['PB1', 'PB2', 'PB3', 'Facility Wide']

def init_db():
    """Initializes all database tables, indexes, and default administrative data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Company Settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS company_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Shifts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS shifts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shift_name TEXT NOT NULL UNIQUE,
        start_time TEXT NOT NULL,       -- HH:MM (24-hour format)
        end_time TEXT NOT NULL,         -- HH:MM (24-hour format)
        grace_period_minutes INTEGER DEFAULT 15,
        is_default INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3. Admin Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admin_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT DEFAULT 'admin',      -- 'admin', 'supervisor', 'hr'
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    """)

    # 4. Employees table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        department TEXT NOT NULL,       -- 'Managers', 'Supervisors', 'Employees', 'Workers', 'Security', 'Others'
        production_block TEXT DEFAULT 'PB1', -- 'PB1', 'PB2', 'PB3', 'Facility Wide'
        designation TEXT NOT NULL,
        contact_number TEXT NOT NULL,
        joining_date TEXT NOT NULL,     -- YYYY-MM-DD
        shift_id INTEGER,
        photo_path TEXT,
        status TEXT DEFAULT 'active',   -- 'active', 'inactive'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (shift_id) REFERENCES shifts(id) ON DELETE SET NULL
    )
    """)

    # Safe migration: ensure production_block exists if table was created previously
    cursor.execute("PRAGMA table_info(employees)")
    emp_cols = {col['name'] for col in cursor.fetchall()}
    if "production_block" not in emp_cols:
        cursor.execute("ALTER TABLE employees ADD COLUMN production_block TEXT DEFAULT 'PB1'")

    # Indexes on employees
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emp_id ON employees(employee_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emp_dept ON employees(department)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emp_block ON employees(production_block)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emp_status ON employees(status)")

    # Data hygiene migration: sanitize department values to strictly the allowed 6 departments
    cursor.execute("""
        UPDATE employees 
        SET department = 'Workers' 
        WHERE department NOT IN ('Managers', 'Supervisors', 'Employees', 'Workers', 'Security', 'Others')
    """)
    cursor.execute("""
        UPDATE employees 
        SET production_block = 'PB1' 
        WHERE production_block IS NULL OR production_block = ''
    """)

    # 5. Employee QR Tokens table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employee_qr_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        token_hash TEXT UNIQUE NOT NULL,
        raw_token_preview TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        revoked_at TIMESTAMP,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_qr_token_hash ON employee_qr_tokens(token_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_qr_emp_id ON employee_qr_tokens(employee_id)")

    # 6. Attendance table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        attendance_date TEXT NOT NULL,          -- YYYY-MM-DD
        check_in_time TEXT,                     -- HH:MM:SS
        check_out_time TEXT,                    -- HH:MM:SS
        total_working_minutes INTEGER DEFAULT 0,
        tea_break_start TEXT,                   -- HH:MM:SS
        tea_break_end TEXT,                     -- HH:MM:SS
        tea_break_minutes INTEGER DEFAULT 0,
        lunch_break_start TEXT,                 -- HH:MM:SS
        lunch_break_end TEXT,                   -- HH:MM:SS
        lunch_break_minutes INTEGER DEFAULT 0,
        total_break_minutes INTEGER DEFAULT 0,
        effective_working_minutes INTEGER DEFAULT 0,
        current_activity TEXT DEFAULT 'Checked-In', -- 'Checked-In', 'Tea Break', 'Lunch Break', 'Shift Closed'
        attendance_method TEXT NOT NULL,        -- 'Face Recognition', 'QR Code', 'Manual/Admin'
        status TEXT DEFAULT 'Present',          -- 'Present', 'Late', 'Half-Day', 'Absent'
        is_late INTEGER DEFAULT 0,
        is_early_departure INTEGER DEFAULT 0,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (employee_id, attendance_date),
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(attendance_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_emp_date ON attendance(employee_id, attendance_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_status ON attendance(status)")

    # Safe migration: ensure break columns exist if table was created previously
    cursor.execute("PRAGMA table_info(attendance)")
    existing_cols = {col['name'] for col in cursor.fetchall()}
    break_cols = [
        ("tea_break_start", "TEXT"),
        ("tea_break_end", "TEXT"),
        ("tea_break_minutes", "INTEGER DEFAULT 0"),
        ("lunch_break_start", "TEXT"),
        ("lunch_break_end", "TEXT"),
        ("lunch_break_minutes", "INTEGER DEFAULT 0"),
        ("total_break_minutes", "INTEGER DEFAULT 0"),
        ("effective_working_minutes", "INTEGER DEFAULT 0"),
        ("current_activity", "TEXT DEFAULT 'Checked-In'")
    ]
    for col_name, col_type in break_cols:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE attendance ADD COLUMN {col_name} {col_type}")

    # 7. Detailed Break Logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS break_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        attendance_id INTEGER,
        attendance_date TEXT NOT NULL,
        break_type TEXT NOT NULL,       -- 'Tea Break', 'Lunch Break'
        start_time TEXT NOT NULL,       -- HH:MM:SS
        end_time TEXT,                  -- HH:MM:SS
        duration_minutes INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (attendance_id) REFERENCES attendance(id) ON DELETE CASCADE,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_break_emp_date ON break_logs(employee_id, attendance_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_break_att_id ON break_logs(attendance_id)")

    # 8. Attendance Audit Logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        attendance_id INTEGER,
        action TEXT NOT NULL,           -- 'CREATE', 'UPDATE', 'DELETE', 'CHECK_IN', 'CHECK_OUT', 'MANUAL_OVERRIDE'
        changed_by TEXT NOT NULL,       -- Admin username
        reason TEXT NOT NULL,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_emp ON attendance_audit_logs(employee_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON attendance_audit_logs(created_at)")

    # Insert default shifts if table empty
    cursor.execute("SELECT COUNT(*) as cnt FROM shifts")
    if cursor.fetchone()['cnt'] == 0:
        cursor.execute("""
        INSERT INTO shifts (shift_name, start_time, end_time, grace_period_minutes, is_default)
        VALUES 
        ('General Shift (09:00 - 17:00)', '09:00', '17:00', 15, 1),
        ('Morning Shift (08:00 - 16:00)', '08:00', '16:00', 15, 0),
        ('Evening Shift (14:00 - 22:00)', '14:00', '22:00', 15, 0),
        ('Night Shift (20:00 - 04:00)', '20:00', '04:00', 15, 0)
        """)

    # Insert/update default company settings
    default_settings = [
        ('company_name', 'Himalaya Wellness Company'),
        ('work_hours_full_day', '8'),
        ('work_hours_half_day', '4'),
        ('timezone', 'Asia/Kolkata'),
        ('allow_qr_checkin', '1'),
        ('allow_face_checkin', '1')
    ]
    for key, val in default_settings:
        cursor.execute("INSERT INTO company_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, val))

    # Personal Administrator Account for Darshan Baliger
    darshan_pwd_hash = generate_password_hash("Darshan@2026")
    cursor.execute("SELECT id FROM admin_users WHERE username = 'darshan'")
    if not cursor.fetchone():
        cursor.execute("""
        INSERT INTO admin_users (username, password_hash, full_name, role, is_active)
        VALUES ('darshan', ?, 'Darshan Baliger', 'admin', 1)
        """, (darshan_pwd_hash,))
    else:
        cursor.execute("UPDATE admin_users SET password_hash = ?, is_active = 1 WHERE username = 'darshan'", (darshan_pwd_hash,))

    # Deactivate or remove generic admin as requested
    cursor.execute("DELETE FROM admin_users WHERE username = 'admin'")

    conn.commit()
    conn.close()
    print("Database initialized successfully with Himalaya Wellness Company profile and Darshan Baliger administrator account.")

if __name__ == '__main__':
    init_db()
