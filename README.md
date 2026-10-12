# Himalaya Wellness Company - Smart Attendance Management System

An enterprise-grade, modern full-stack web application designed for companies and manufacturing plants to streamline employee registration, daily attendance tracking, check-in and check-out management, biometric face recognition, secure QR code verification, shift breaks (Tea & Lunch), production block division (PB1, PB2, PB3), working hours telemetry, and corporate reporting.

---

## 🌟 Key Features

1. **Executive Operations Dashboard**
   - Live metrics: Total Registered Employees, Present Today, Absent Today, Active on Duty (Checked-In), Checked-Out, and Today's Attendance Percentage.
   - **Production Block Breakdown:** Real-time presence telemetry across Himalaya Wellness's 3 major production blocks:
     - **`PB1`** (Tablets & Capsules Formulation)
     - **`PB2`** (Liquids & Syrups Processing)
     - **`PB3`** (Ointments, Creams & Packaging)
     - **`Facility Wide`** (Plant Security, Facilities & Administration)
   - Real-time attendance feed showing live check-ins and check-outs with timestamps and methods.
   - Department-wise presence telemetry across the strict corporate hierarchy: `Managers`, `Supervisors`, `Employees`, `Workers`, `Security`, `Others`.

2. **Employee Registration & Biometric Enrollment**
   - Fields: Unique Employee ID, Full Name, Department, Production Block (PB1/PB2/PB3/Facility Wide), Designation, Contact Number, Date of Joining, Assigned Work Shift.
   - Dual Photo Capture: Direct file upload or live camera capture via webcam modal.
   - Automated biometric facial feature extraction (128D embedding) using OpenCV YuNet face detection and SFace deep learning models.
   - Automatic generation of a cryptographically secure, revocable QR Code badge with SHA-256 token hashing.
   - Strict avoidance of email/OTP dependencies for direct corporate onboarding.

3. **Face Recognition Attendance Kiosk**
   - Uses **YuNet** (`cv2.FaceDetectorYN`) for face detection and **SFace** (`cv2.FaceRecognizerSF`) for recognition.
   - Real-time browser webcam streaming via `getUserMedia` + canvas capture.
   - Cosine similarity matching (threshold ≥ 0.38) with calibrated confidence percentages.
   - **Interactive Modal Feedback:** Displays a Blue Tick Mark confirmation dialog with break durations and net hours.
   - **Automatic Camera Deactivation:** Automatically closes the webcam when an employee completes their work shift or attempts duplicate scans.

4. **Break Tracking & Shift Close Workflow**
   - **☕ Tea Break:** Quick 15-minute scheduled break tracker (logs start and end timestamps).
   - **🍱 Lunch Break:** Mid-day meal break tracker (logs start and end timestamps).
   - **🚪 Shift Close:** Final day checkout that calculates **Net Effective Productive Hours** (Total Session Time minus Total Break Minutes).
   - Prevents duplicate attendance logging once a shift is finalized.

5. **Cryptographically Sealed QR Code Attendance**
   - Generates unique unguessable tokens (`EMPQR_<id>_<token>`) stored as SHA-256 hashes in SQLite.
   - High-speed real-time camera QR scanning via pure client-side `jsQR` (100% offline capable).
   - Server-side fallback scanner decoding via OpenCV `cv2.QRCodeDetector` for image file uploads.
   - Duplicate protection and instant check-in / check-out resolution.

6. **Employee Directory & Lifecycle Management**
   - Multi-parameter search and filtering by Employee ID, Name, Department, Production Block, and Employment Status.
   - Employee detail views, printable QR badge downloads, and token revocation / regeneration.
   - Quick Edit modal to update profile details, work shift, or assigned production block.
   - Safe deactivation / reactivation toggle.
   - **Permanent Delete Action:** Fully cleanses employee profile, photo, 128D face embeddings from RAM/disk, QR badges, and historical logs with audit verification.

7. **Reports & Analytics Engine**
   - Daily, Weekly, and Monthly attendance reports.
   - **Production Block Summary:** Aggregates headcount, present counts, and attendance percentage per PB1, PB2, PB3.
   - Department-wise and Employee-wise cumulative work hours summaries.
   - Late arrival and absence reports with exact time discrepancies.
   - Direct export of filtered attendance datasets into downloadable CSV files.

8. **Security & Governance**
   - Dedicated private administrator credentials.
   - CSRF protection on all state-changing forms and API requests.
   - In-memory login rate-limiting (temporary 15-minute lockout after 5 consecutive failed attempts).
   - Zero SQL injection risks via parameterized SQLite queries.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.10+ / 3.14+, Flask 3.x, Werkzeug, Gunicorn
- **Biometrics:** OpenCV (`opencv-python-headless`), YuNet ONNX, SFace ONNX
- **QR Codes:** Python `qrcode` library, `jsQR` client-side library, OpenCV `QRCodeDetector`
- **Database:** SQLite 3 (Normalized relational schema with foreign keys, indexes, and serverless `/tmp` fallback)
- **Frontend:** Modern Semantic HTML5, Vanilla CSS3 (Corporate Dark-Navy theme), JavaScript ES6+ (Web Audio API, WebRTC)

---

## 📁 Project Structure

```text
smart-attendance-system/
├── app.py                      # Master Flask application & API routes
├── db.py                       # SQLite database schema, connections & seed data
├── models_utils.py             # OpenCV YuNet & SFace deep learning wrapper
├── create_face_database.py     # CLI tool to rebuild/verify face embeddings
├── test_system.py              # Automated 10-step end-to-end verification suite
├── requirements.txt            # Python dependencies
├── vercel.json                 # Vercel Serverless deployment configuration
├── Procfile                    # WSGI server process file (Render / Railway)
├── wsgi.py                     # WSGI entrypoint for production servers
├── .gitignore                  # Git exclusion rules for private data & db
├── README.md                   # System documentation & deployment guide
├── database/
│   ├── .gitkeep
│   └── attendance.db           # SQLite database file (created upon first run)
├── models/
│   ├── face_detection_yunet_2023mar.onnx     # YuNet Face Detector
│   └── face_recognition_sface_2021dec.onnx   # SFace Face Recognizer
├── face_data/                  # Serialized .npy face embeddings
├── qr_codes/                   # Generated employee QR badge PNGs
├── templates/
│   ├── base.html               # Corporate layout, sidebar & live clock
│   ├── login.html              # Admin authentication portal
│   ├── index.html              # Executive telemetry dashboard
│   ├── register_employee.html  # Employee onboarding & photo enrollment
│   ├── employees.html          # Employee directory & roster
│   ├── employee_details.html   # Profile, QR badge card & employee log
│   ├── face_attendance.html    # Face recognition kiosk interface
│   ├── scan_qr.html            # QR scanner kiosk interface
│   ├── attendance.html         # Attendance logs & admin correction modal
│   ├── reports.html            # Reports & analytics engine
│   └── settings.html           # Shift schedules & company configuration
└── static/
    ├── css/
    │   └── style.css           # Master CSS design system
    ├── js/
    │   ├── script.js           # Client utilities, toast & audio feedback
    │   ├── face_scanner.js     # Real-time face recognition camera logic
    │   ├── qr_scanner.js       # Real-time QR scanner camera logic
    │   └── jsQR.js             # Offline client-side QR decoder
    └── images/
        ├── default-avatar.png  # Profile placeholder
        └── employees/          # Enrolled employee photos
```

---

## 🔑 Administrator Credentials

- **Portal URL:** `http://127.0.0.1:5000/login`
- **Username:** `darshan`
- **Password:** `Darshan@2026`

---

## 🚀 Local Development Setup

### 1. Open Terminal in VS Code
Press ``Ctrl + ` `` in VS Code:
```powershell
cd "c:\Users\DARSHAN BALIGER\OneDrive\Desktop\smart-attendance system"
```

### 2. Run Database & Seed Initialization
```powershell
py db.py
```

### 3. Run Automated System Tests
```powershell
py test_system.py
```

### 4. Start Local Development Server
```powershell
py app.py
```
Navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🌐 Step-by-Step Deployment Guide: GitHub & Vercel

### Part 1: Push Code to GitHub

#### Step 1: Create a New GitHub Repository
1. Open your web browser and go to [https://github.com](https://github.com).
2. Sign in to your GitHub account.
3. Click the **`+`** icon in the top-right corner and select **New repository**.
4. Set Repository Name: `smart-attendance-system` (or any preferred name).
5. Choose **Public** or **Private**.
6. **IMPORTANT:** Do **NOT** check "Add a README file", "Add .gitignore", or "Choose a license" (we already have them configured in the project).
7. Click **Create repository**.
8. Copy your repository's URL (e.g., `https://github.com/YOUR_USERNAME/smart-attendance-system.git`).

#### Step 2: Initialize Git and Commit Your Project
In your project directory terminal in VS Code, run:
```powershell
# 1. Ensure you are on the main branch
git branch -M main

# 2. Stage all project files (respecting .gitignore)
git add .

# 3. Create initial commit
git commit -m "Initial commit: Production Smart Attendance System with Face Biometrics and Production Blocks"

# 4. Link your local repository to your GitHub repository (replace with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/smart-attendance-system.git

# 5. Push to GitHub
git push -u origin main
```

---

### Part 2: Deploy to Vercel

#### Step 1: Sign In to Vercel
1. Go to [https://vercel.com](https://vercel.com).
2. Log in with your **GitHub** account.

#### Step 2: Import Your Repository
1. On your Vercel Dashboard, click **Add New...** → **Project**.
2. Find your `smart-attendance-system` repository and click **Import**.

#### Step 3: Configure Project Settings
1. **Framework Preset:** Select **Other** (or leave as default).
2. **Root Directory:** `./` (leave default).
3. **Environment Variables:** Add the following environment variable:
   - **Key:** `SECRET_KEY`
   - **Value:** `your-super-secret-random-key-2026`
4. Click **Deploy**.

Vercel will detect `vercel.json` and `requirements.txt`, install dependencies, package the Python application into a serverless function, and assign a live production URL (e.g., `https://smart-attendance-system.vercel.app`).

> [!NOTE]
> **Understanding Serverless Hosting (Vercel):**
> Vercel executes applications inside stateless serverless containers where the root directory is read-only. Our project includes automated `/tmp` fallback (`/tmp/attendance.db` and `/tmp/face_data/`) so the app runs smoothly on Vercel. However, `/tmp` in serverless is ephemeral (data resets when containers cycle).

---

### Part 3: Recommended Persistent Cloud Deployment (Render / Railway)

For a production factory environment with persistent SQLite database writes, employee photos, and face embeddings, deploying to **Render** or **Railway** gives you a free persistent cloud server:

#### Deploying on Render (Free, Persistent, 1-Click):
1. Go to [https://render.com](https://render.com) and sign in with GitHub.
2. Click **New +** → **Web Service**.
3. Select your GitHub repository `smart-attendance-system`.
4. Configure:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click **Create Web Service**.
6. Render builds the app with Gunicorn and provides a free **`https://your-app.onrender.com`** URL with automatic HTTPS (enabling webcam streaming globally).

---

## 📹 Webcam Access & Browser Permissions

1. **HTTPS Requirement:** Modern browsers (Chrome, Edge, Firefox, Safari) only permit webcam access (`navigator.mediaDevices.getUserMedia`) on **`localhost`** or secure **`https://`** domains.
2. **Permission Prompt:** When visiting **Face Attendance** (`/face_attendance`) or **QR Scanner** (`/scan_qr`), click **"Allow"** when prompted for camera access.
3. **If Camera is Blocked:** Click the camera lock icon in your browser URL bar, change permission to "Allow", and reload the page.
