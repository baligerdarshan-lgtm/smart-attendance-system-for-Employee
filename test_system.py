"""
Automated End-to-End Verification Test Script
Tests database, face engine, QR token engine, check-in/out workflows, Flask routes, and audit trails.
"""

import os
import sys
import json
import sqlite3
import numpy as np

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import app components
import db
import app as flask_app
from models_utils import FaceEngine

def run_tests():
    print("=" * 70)
    print(" Smart Attendance System - Automated Verification Suite")
    print("=" * 70)

    # 1. Test Database
    print("\n[TEST 1] Verifying SQLite Database Schema & Seeds...")
    db.init_db()
    conn = db.get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as cnt FROM shifts")
    shift_count = cursor.fetchone()['cnt']
    assert shift_count >= 4, f"Expected at least 4 default shifts, found {shift_count}"
    print(f" [PASS] Shifts table verified: {shift_count} configured shifts.")

    cursor.execute("SELECT COUNT(*) as cnt FROM admin_users WHERE username = 'darshan'")
    admin_count = cursor.fetchone()['cnt']
    assert admin_count == 1, "Administrator account 'darshan' not found"
    print(" [PASS] Darshan Baliger admin account verified.")
    conn.close()

    # 2. Test Face Engine & Models
    print("\n[TEST 2] Verifying OpenCV YuNet & SFace Neural Engine...")
    engine = FaceEngine.get_instance()
    assert engine.detector is not None, "YuNet detector failed to initialize"
    assert engine.recognizer is not None, "SFace recognizer failed to initialize"
    print(f" [PASS] YuNet & SFace models loaded cleanly.")

    # 3. Test Secure QR Generation & Hash
    print("\n[TEST 3] Verifying Secure QR Token Cryptography...")
    raw_tok, tok_hash = db.generate_secure_qr_token("EMP-9999")
    assert raw_tok.startswith("EMPQR_EMP-9999_"), f"Unexpected raw token format: {raw_tok}"
    assert db.hash_qr_token(raw_tok) == tok_hash, "Hash verification failed"
    print(f" [PASS] Generated Token: {raw_tok[:20]}... Hash: {tok_hash[:16]}...")

    # 4. Test Flask Routes & Session
    print("\n[TEST 4] Testing Flask Client Routes & Security...")
    flask_app.app.testing = True
    client = flask_app.app.test_client()

    # Test login page
    res = client.get('/login')
    assert res.status_code == 200, f"GET /login returned {res.status_code}"
    print(" [PASS] GET /login: 200 OK")

    # Perform admin login as darshan
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['username'] = 'darshan'
        sess['full_name'] = 'Darshan Baliger'
        sess['role'] = 'admin'
        csrf_tok = db.generate_secure_qr_token("CSRF")[1]
        sess['csrf_token'] = csrf_tok

    # Test authenticated routes
    endpoints = [
        ('/', 200),
        ('/dashboard', 200),
        ('/register', 200),
        ('/employees', 200),
        ('/face_attendance', 200),
        ('/scan_qr', 200),
        ('/attendance', 200),
        ('/reports', 200),
        ('/settings', 200),
        ('/reports/export', 200)
    ]
    for path, exp_code in endpoints:
        r = client.get(path)
        assert r.status_code == exp_code, f"GET {path} expected {exp_code}, got {r.status_code}"
        print(f" [PASS] GET {path}: {r.status_code} OK")

    # 5. Test Employee Registration & Full Attendance Cycle (Check-In -> Tea Break -> Lunch Break -> Shift Close)
    print("\n[TEST 5] Testing Full Lifecycle: Check-In -> Tea Break -> Lunch Break -> Shift Close...")
    test_emp_id = "EMP-TEST01"
    
    # Remove previous test record if exists
    conn = db.get_db_connection()
    conn.execute("DELETE FROM break_logs WHERE employee_id = ?", (test_emp_id,))
    conn.execute("DELETE FROM attendance WHERE employee_id = ?", (test_emp_id,))
    conn.execute("DELETE FROM employee_qr_tokens WHERE employee_id = ?", (test_emp_id,))
    conn.execute("DELETE FROM employees WHERE employee_id = ?", (test_emp_id,))
    conn.commit()

    # Insert test employee
    # Insert test employee with strict department and production block
    raw_tok, tok_hash = db.generate_secure_qr_token(test_emp_id)
    conn.execute("""
        INSERT INTO employees (employee_id, full_name, department, production_block, designation, contact_number, joining_date, shift_id, status)
        VALUES (?, 'Test Employee Alpha', 'Supervisors', 'PB2', 'Block Supervisor', '9998887770', '2026-01-01', 1, 'active')
    """, (test_emp_id,))
    conn.execute("""
        INSERT INTO employee_qr_tokens (employee_id, token_hash, raw_token_preview, is_active)
        VALUES (?, ?, 'EMPQR_PREVIEW...', 1)
    """, (test_emp_id, tok_hash))
    conn.commit()
    conn.close()
    print(f" [PASS] Test employee '{test_emp_id}' created with Department='Supervisors' and Block='PB2'.")

    # 5a. Check-In via QR API
    qr_res = client.post('/api/scan_qr', json={'token': raw_tok, 'event_type': 'check_in'}, headers={'X-CSRFToken': csrf_tok})
    assert qr_res.status_code == 200, f"QR scan returned {qr_res.status_code}: {qr_res.data}"
    qr_data = qr_res.get_json()
    assert qr_data['success'] is True, f"Check-in failed: {qr_data}"
    assert qr_data['attendance']['action'] == 'check_in', "Expected check-in action"
    assert qr_data['employee']['production_block'] == 'PB2', "Expected production block PB2"
    print(f" [PASS] 1. Check-In event successfully logged: {qr_data['message']}")

    # 5b. Tea Break Start
    tea_start_res = flask_app.process_attendance_event(test_emp_id, "QR Code", event_type="tea_break")
    assert tea_start_res['success'] is True, f"Tea break start failed: {tea_start_res}"
    assert tea_start_res['action'] == 'tea_break_start', "Expected tea_break_start"
    print(f" [PASS] 2. Tea Break Started: {tea_start_res['message']}")

    # 5c. Tea Break End (Resuming duty)
    tea_end_res = flask_app.process_attendance_event(test_emp_id, "QR Code", event_type="tea_break")
    assert tea_end_res['success'] is True, f"Tea break end failed: {tea_end_res}"
    assert tea_end_res['action'] == 'tea_break_end', "Expected tea_break_end"
    print(f" [PASS] 3. Tea Break Ended: {tea_end_res['message']}")

    # 5d. Lunch Break Start
    lunch_start_res = flask_app.process_attendance_event(test_emp_id, "QR Code", event_type="lunch_break")
    assert lunch_start_res['success'] is True, f"Lunch break start failed: {lunch_start_res}"
    assert lunch_start_res['action'] == 'lunch_break_start', "Expected lunch_break_start"
    print(f" [PASS] 4. Lunch Break Started: {lunch_start_res['message']}")

    # 5e. Lunch Break End (Resuming duty)
    lunch_end_res = flask_app.process_attendance_event(test_emp_id, "QR Code", event_type="lunch_break")
    assert lunch_end_res['success'] is True, f"Lunch break end failed: {lunch_end_res}"
    assert lunch_end_res['action'] == 'lunch_break_end', "Expected lunch_break_end"
    print(f" [PASS] 5. Lunch Break Ended: {lunch_end_res['message']}")

    # 5f. Final Shift Close (Check-Out)
    shift_close_res = flask_app.process_attendance_event(test_emp_id, "QR Code", event_type="check_out")
    assert shift_close_res['success'] is True, f"Shift close failed: {shift_close_res}"
    assert shift_close_res['action'] == 'check_out', "Expected check_out"
    print(f" [PASS] 6. Final Shift Closed (Check-Out): {shift_close_res['message']}")

    # 5g. Verify Break Logs recorded in database
    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM break_logs WHERE employee_id = ?", (test_emp_id,))
    break_log_count = cursor.fetchone()['count']
    assert break_log_count == 2, f"Expected 2 break logs (Tea & Lunch), found {break_log_count}"
    print(f" [PASS] 7. Verified {break_log_count} break logs in break_logs table.")

    # 5h. Duplicate scan after shift is closed -> prevented & returns error with webcam close signal
    dup_res = flask_app.process_attendance_event(test_emp_id, "QR Code")
    assert dup_res['success'] is False, "Expected error on duplicate attempt after shift close"
    assert dup_res.get('is_completed') is True, "Expected is_completed flag for automatic camera close"
    print(f" [PASS] 8. Duplicate attendance prevented after shift close: {dup_res['error']}")

    # 5i. Test block_wise and production_block filters on reports & CSV
    block_rep = client.get('/reports?report_type=block_wise&production_block=PB2')
    assert block_rep.status_code == 200, f"Block report returned {block_rep.status_code}"
    csv_rep = client.get('/reports/export?report_type=block_wise&production_block=PB2')
    assert csv_rep.status_code == 200, f"CSV export returned {csv_rep.status_code}"
    print(" [PASS] 9. Block-wise report & CSV export verified.")

    # 5j. Test Delete Employee Route & Complete Cascading Cleanup
    del_res = client.post(f'/employees/{test_emp_id}/delete', data={'csrf_token': csrf_tok}, follow_redirects=True)
    assert del_res.status_code == 200, f"Delete employee returned {del_res.status_code}"
    
    # Verify employee and child records are completely gone
    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM employees WHERE employee_id = ?", (test_emp_id,))
    assert cursor.fetchone()['count'] == 0, "Employee record still exists after deletion"
    cursor.execute("SELECT COUNT(*) as count FROM attendance WHERE employee_id = ?", (test_emp_id,))
    assert cursor.fetchone()['count'] == 0, "Attendance records still exist after employee deletion"
    cursor.execute("SELECT COUNT(*) as count FROM break_logs WHERE employee_id = ?", (test_emp_id,))
    assert cursor.fetchone()['count'] == 0, "Break logs still exist after employee deletion"
    cursor.execute("SELECT COUNT(*) as count FROM employee_qr_tokens WHERE employee_id = ?", (test_emp_id,))
    assert cursor.fetchone()['count'] == 0, "QR tokens still exist after employee deletion"
    conn.close()
    print(" [PASS] 10. Delete employee endpoint verified with full cascading purge.")

    print("\n" + "=" * 70)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()
