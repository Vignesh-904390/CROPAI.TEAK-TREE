import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, url_for, redirect, flash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = 'f3956c777cebc566ffb95408917364c2'

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/teak_effnet.pth'            # <-- TEAK model path
DISEASE_JSON_PATH = 'teak_disease_info.json'    # <-- TEAK disease JSON
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Mail Configuration ===
# (keep these but replace with your actual credentials if needed)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'v9630094@gmail.com'
app.config['MAIL_PASSWORD'] = 'rtwt opco zptt jwvz'
mail = Mail(app)

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}")

# === Load Model (checkpoint must contain 'model_state_dict' and 'class_names') ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please place your TEAK .pth checkpoint there.")

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' not in checkpoint or 'class_names' not in checkpoint:
    raise KeyError("Checkpoint must contain 'model_state_dict' and 'class_names' keys.")

class_names = checkpoint['class_names']

# Use the modern torchvision API - weights=None avoids deprecated 'pretrained' arg
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def normalize_key(name):
    return ''.join(e.lower() for e in name.strip() if e.isalnum())

# === Load disease details with UTF-8 encoding (create minimal placeholder if missing) ===
if not os.path.exists(DISEASE_JSON_PATH):
    placeholder = {
        "Healthy": {
            "fertilizer": "Maintain balanced nutrients",
            "water": "Normal field level",
            "medicine": ["No chemical treatment needed"],
            "organic_medicine": ["Neem cake", "Vermicompost"],
            "prevention": "Continue regular crop monitoring and hygiene."
        }
    }
    with open(DISEASE_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(placeholder, fp, ensure_ascii=False, indent=4)
    print(f"⚠️ Created placeholder disease JSON at {DISEASE_JSON_PATH} — please replace with full TEAK entries.")

with open(DISEASE_JSON_PATH, 'r', encoding='utf-8') as f:
    raw_disease_details = json.load(f)
disease_details = { normalize_key(k): v for k, v in raw_disease_details.items() }

REGION_GRID = (2, 2)  # split each image into 4 regions for more robust classification

# === Daily stats & reporting ===
daily_stats = {"count": 0, "timestamps": []}

def log_click():
    daily_stats["count"] += 1
    daily_stats["timestamps"].append(datetime.now().strftime("%H:%M:%S"))

def send_daily_report():
    if daily_stats["count"] == 0:
        return
    try:
        msg = Message("📊 Daily Click Report - Teak Disease Detection",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=['tdaitech@gmail.com'])
        times = "\n".join(daily_stats["timestamps"])
        msg.body = f"Total Clicks Today: {daily_stats['count']}\n\nTimes:\n{times}"
        mail.send(msg)
        # reset counters
        daily_stats["count"] = 0
        daily_stats["timestamps"] = []
    except Exception as e:
        print("❌ Error sending daily report:", e)

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(send_daily_report, 'cron', hour=23, minute=59)
scheduler.start()

# === Utility: split image into grid regions ===
def split_image_regions(image, grid=(2,2)):
    w, h = image.size
    ws, hs = w // grid[0], h // grid[1]
    regions = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            left, top = i * ws, j * hs
            # ensure last tile reaches the edge (avoid rounding issues)
            right = left + ws if (i < grid[0] - 1) else w
            bottom = top + hs if (j < grid[1] - 1) else h
            regions.append(image.crop((left, top, right, bottom)))
    return regions

# === Routes ===
@app.route('/')
def index():
    # index.html should exist in templates/ — keep same interface as your previous apps
    return render_template('index.html', title="Teak Disease Detection")

@app.route('/predict', methods=['POST'])
def predict_image():
    log_click()
    if 'image' not in request.files or request.files['image'].filename == '':
        flash("❌ No image uploaded", "danger")
        return redirect('/')
    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = Image.open(path).convert('RGB')
    subs = split_image_regions(img, REGION_GRID)

    best_conf = 0.0
    best_label = None
    for r in subs:
        t = transform(r).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_label = class_names[pred.item()].strip()

    if not best_label:
        flash("⚠️ Could not classify image", "warning")
        return redirect('/')

    label = "Healthy" if best_label.lower() == "healthy" else best_label
    d = disease_details.get(normalize_key(label), {})

    info = [{
        "label": label,
        "details": {
            "explanation": d.get("explanation", f"Detected {label}."),
            "water": d.get("water", "N/A"),
            "fertilizer": d.get("fertilizer", "N/A"),
            "medicine": d.get("medicine", ["N/A"]),
            "organic_medicine": d.get("organic_medicine", ["N/A"]),
            "prevention": d.get("prevention", "N/A")
        }
    }]

    return render_template('index.html', multi_predictions=info,
                           image_url=url_for('static', filename='uploads/' + filename),
                           title="Teak Disease Detection")

@app.route('/predict_video', methods=['POST'])
def predict_video():
    log_click()
    if 'video' not in request.files or request.files['video'].filename == '':
        flash("❌ No video uploaded", "danger")
        return redirect('/')
    f = request.files['video']
    name = secure_filename(f.filename)
    vp = os.path.join(UPLOAD_FOLDER, name)
    f.save(vp)

    cap = cv2.VideoCapture(vp)
    fr = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fr) if fr > 0 else 10

    preds = []
    i = 0
    while cap.isOpened():
        r, frm = cap.read()
        if not r:
            break
        if i % interval == 0:
            g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            if 40 < np.mean(g) < 220:
                pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                for s in split_image_regions(pil, REGION_GRID):
                    t = transform(s).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(t)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        _, pr = torch.max(probs, 1)
                        preds.append(class_names[pr.item()].strip())
        i += 1

    cap.release()
    try:
        os.remove(vp)
    except Exception:
        pass

    if not preds:
        flash("⚠️ No disease found", "warning")
        return redirect('/')

    c = Counter(preds)
    mc = [l for l, n in c.items() if n >= 2] or list(c.keys())
    if any(l.lower() == "healthy" for l in mc):
        mc = ["Healthy"]

    info = []
    for l in mc:
        d = disease_details.get(normalize_key(l), {})
        info.append({"label": l, "details": {
            "explanation": d.get("explanation", f"Detected {l}."),
            "water": d.get("water", "N/A"),
            "fertilizer": d.get("fertilizer", "N/A"),
            "medicine": d.get("medicine", ["N/A"]),
            "organic_medicine": d.get("organic_medicine", ["N/A"]),
            "prevention": d.get("prevention", "N/A")
        }})

    return render_template('index.html', multi_predictions=info, image_url=None, title="Teak Disease Detection")

@app.route('/send_email', methods=['POST'])
def send_email():
    log_click()
    name = request.form.get('name')
    email = request.form.get('email')
    msgt = request.form.get('message')
    photo = request.files.get('photo')

    if not (name and email and msgt):
        flash("❗ Fill all fields", "warning")
        return redirect('/')

    try:
        m = Message("🌿 New Contact Request - Teak Detection",
                    sender=app.config['MAIL_USERNAME'],
                    recipients=['tdaitech@gmail.com'])
        m.body = f"Name:{name}\nEmail:{email}\nMessage:{msgt}"
        if photo and photo.filename:
            fn = secure_filename(photo.filename)
            fp = os.path.join(UPLOAD_FOLDER, fn)
            photo.save(fp)
            with open(fp, 'rb') as fh:
                m.attach(fn, "image/jpeg", fh.read())
        mail.send(m)

        r = Message("✅ Thank you!", sender=app.config['MAIL_USERNAME'], recipients=[email])
        r.body = f"Hi {name},\nWe received your message."
        mail.send(r)
        flash("✅ Message sent!", "success")
    except Exception as e:
        print("❌ Error sending contact/email:", e)
        flash("❌ Failed to send", "danger")
    return redirect('/')

# === Run ===
if __name__ == "__main__":
    # set debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
