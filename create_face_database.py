"""
CLI utility to re-index, rebuild, or verify the face embeddings database from registered employee photos.
Can be executed standalone: python create_face_database.py
"""

import os
import cv2
import sqlite3
from models_utils import FaceEngine, BASE_DIR
from db import get_db_connection

def rebuild_face_database():
    print("=" * 60)
    print(" Smart Attendance System - Face Database Rebuilder")
    print("=" * 60)

    engine = FaceEngine.get_instance()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT employee_id, full_name, photo_path, status FROM employees")
    employees = cursor.fetchall()
    print(f"Found {len(employees)} employee record(s) in SQLite database.")

    success_count = 0
    fail_count = 0

    for emp in employees:
        emp_id = emp['employee_id']
        name = emp['full_name']
        photo_rel = emp['photo_path']

        if not photo_rel:
            print(f"[-] {emp_id} ({name}): No photo path registered.")
            fail_count += 1
            continue

        photo_abs = os.path.join(BASE_DIR, photo_rel.lstrip('/\\'))
        if not os.path.exists(photo_abs):
            print(f"[-] {emp_id} ({name}): Photo file not found at {photo_abs}")
            fail_count += 1
            continue

        img = cv2.imread(photo_abs)
        if img is None:
            print(f"[-] {emp_id} ({name}): Failed to read image file.")
            fail_count += 1
            continue

        success, msg = engine.register_employee_face(emp_id, img)
        if success:
            print(f"[+] {emp_id} ({name}): Successfully processed and saved feature embedding.")
            success_count += 1
        else:
            print(f"[-] {emp_id} ({name}): Face feature extraction failed - {msg}")
            fail_count += 1

    conn.close()
    engine.load_all_embeddings()
    print("=" * 60)
    print(f"Summary: {success_count} indexed successfully, {fail_count} failed or skipped.")
    print(f"Total active memory embeddings: {len(engine.embeddings)}")
    print("=" * 60)

if __name__ == '__main__':
    rebuild_face_database()
