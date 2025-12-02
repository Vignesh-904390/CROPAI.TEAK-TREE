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
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'v9630094@gmail.com'
app.config['MAIL_PASSWORD'] = 'rtwt opco zptt jwvz'
mail = Mail(app)

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}")

# === Load Model ===
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("🧠 Teak EfficientNet model loaded.")

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Load Disease Info ===
def normalize_key(name):
    return ''.join(e.lower() for e in name.strip() if e.isalnum())

raw_disease_details = None
_used_encoding = None

# Try utf-8 first, then fall back to cp1252 / latin-1 if needed
try:
    with open(DISEASE_JSON_PATH, 'r', encoding='utf-8') as f:
        raw_disease_details = json.load(f)
        _used_encoding = 'utf-8'
except UnicodeDecodeError:
    try:
        # try with cp1252 (windows default) or latin-1 as a permissive fallback
        with open(DISEASE_JSON_PATH, 'r', encoding='cp1252') as f:
            raw_disease_details = json.load(f)
            _used_encoding = 'cp1252'
    except Exception:
        try:
            with open(DISEASE_JSON_PATH, 'r', encoding='latin-1') as f:
                raw_disease_details = json.load(f)
                _used_encoding = 'latin-1'
        except Exception as e:
            print(f"❌ Failed to load {DISEASE_JSON_PATH}: {e}")
            raw_disease_details = {}

except FileNotFoundError:
    print(f"❌ Disease info file not found: {DISEASE_JSON_PATH}")
    raw_disease_details = {}
except json.JSONDecodeError as e:
    print(f"❌ JSON decode error while reading {DISEASE_JSON_PATH}: {e}")
    raw_disease_details = {}
except Exception as e:
    print(f"❌ Unexpected error loading {DISEASE_JSON_PATH}: {e}")
    raw_disease_details = {}

if _used_encoding:
    print(f"ℹ️ Loaded {DISEASE_JSON_PATH} using encoding: {_used_encoding}")

# Ensure we have a dict (avoid crash if file was empty)
if not isinstance(raw_disease_details, dict):
    raw_disease_details = {}

disease_details = {normalize_key(k): v for k, v in raw_disease_details.items()}

REGION_GRID = (2, 2)  # split image into 2x2 regions

# === Daily stats ===
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
        daily_stats["count"] = 0
        daily_stats["timestamps"] = []
    except Exception as e:
        print("❌ Error sending daily report:", e)

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(send_daily_report, 'cron', hour=23, minute=59)
scheduler.start()

# === Image Split ===
def split_image_regions(image, grid=(2,2)):
    w, h = image.size
    ws, hs = w // grid[0], h // grid[1]
    regions = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            left, top = i*ws, j*hs
            regions.append(image.crop((left, top, left+ws, top+hs)))
    return regions

# === Send Prediction Result Email ===
def send_prediction_result_email(filename, prediction_results, image_path=None):
    try:
        msg = Message("🌳 New Teak Disease Detection Result",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=['tdaitech@gmail.com'])
        
        # Create email body with prediction results
        email_body = f"""
        🔍 Teak Disease Detection Result
        
        📄 File Name: {filename}
        ⏰ Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        📊 PREDICTION RESULTS:
        """
        
        for i, result in enumerate(prediction_results, 1):
            label = result['label']
            details = result['details']
            
            email_body += f"""
            🎯 Result {i}:
            Disease: {label}
            
            📝 Explanation: {details['explanation']}
            💧 Water Requirements: {details['water']}
            🌱 Fertilizer: {details['fertilizer']}
            💊 Medicine: {', '.join(details['medicine'])}
            🌿 Organic Medicine: {', '.join(details['organic_medicine'])}
            🛡️ Prevention: {details['prevention']}
            {'='*50}
            """
        
        msg.body = email_body
        
        # Attach the uploaded image if available
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                msg.attach(filename, "image/jpeg", img_file.read())
        
        mail.send(msg)
        print("✅ Prediction result email sent successfully!")
        
    except Exception as e:
        print("❌ Error sending prediction result email:", e)

# === Extract details from new JSON format ===
def extract_disease_details(disease_data, label):
    """Extract disease details from the new JSON format with both English and Tamil keys"""
    if not disease_data:
        return {
            "explanation": f"Detected {label}.",
            "water": "N/A",
            "fertilizer": "N/A", 
            "medicine": ["N/A"],
            "organic_medicine": ["N/A"],
            "prevention": "N/A"
        }
    
    # Extract English details
    explanation = disease_data.get("explanation", f"Detected {label}.")
    water = disease_data.get("water", "N/A")
    fertilizer = disease_data.get("fertilizer", "N/A")
    medicine = disease_data.get("medicine", ["N/A"])
    organic_medicine = disease_data.get("organic_medicine", ["N/A"])
    prevention = disease_data.get("prevention", "N/A")
    
    # Extract Tamil details
    tamil_explanation = disease_data.get("விளக்கம்", explanation)
    tamil_water = disease_data.get("நீர்", water)
    tamil_fertilizer = disease_data.get("உரம்", fertilizer)
    tamil_medicine = disease_data.get("மருந்து", medicine)
    tamil_organic_medicine = disease_data.get("கரிம மருந்து", organic_medicine)
    tamil_prevention = disease_data.get("தடுப்பு முறைகள்", prevention)
    
    return {
        "explanation": explanation,
        "water": water,
        "fertilizer": fertilizer,
        "medicine": medicine,
        "organic_medicine": organic_medicine,
        "prevention": prevention,
        "tamil_details": {
            "விளக்கம்": tamil_explanation,
            "நீர்": tamil_water,
            "உரம்": tamil_fertilizer,
            "மருந்து": tamil_medicine,
            "கரிம மருந்து": tamil_organic_medicine,
            "தடுப்பு முறைகள்": tamil_prevention
        }
    }

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

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

    best_conf, best_label = 0, None
    for r in subs:
        t = transform(r).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.nn.functional.softmax(model(t), dim=1)
            conf, pred = torch.max(p, 1)
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_label = class_names[pred.item()].strip()

    if not best_label:
        flash("⚠️ Could not classify image", "warning")
        return redirect('/')

    label = "Healthy" if best_label.lower() == "healthy" else best_label
    d = disease_details.get(normalize_key(label), {})

    # Use the new function to extract details from JSON
    disease_info = extract_disease_details(d, label)
    
    info = [{
        "label": label, 
        "details": disease_info
    }]

    # Send prediction result to email
    send_prediction_result_email(filename, info, path)

    return render_template('index.html', multi_predictions=info, image_url=url_for('static', filename='uploads/'+filename))

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

    preds, i = [], 0
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
                        p = torch.nn.functional.softmax(model(t), dim=1)
                        _, pr = torch.max(p, 1)
                        preds.append(class_names[pr.item()].strip())
        i += 1
    cap.release()
    os.remove(vp)

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
        # Use the new function to extract details from JSON
        disease_info = extract_disease_details(d, l)
        info.append({
            "label": l, 
            "details": disease_info
        })

    # Send video prediction result to email
    send_prediction_result_email(name, info)

    return render_template('index.html', multi_predictions=info, image_url=None)

# === Contact Email ===
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
        m = Message("🌳 New Contact Request - Teak Detection", sender=app.config['MAIL_USERNAME'], recipients=['tdaitech@gmail.com'])
        m.body = f"Name:{name}\nEmail:{email}\nMessage:{msgt}"
        if photo and photo.filename:
            fn = secure_filename(photo.filename)
            fp = os.path.join(UPLOAD_FOLDER, fn)
            photo.save(fp)
            with open(fp, 'rb') as f:
                m.attach(fn, "image/jpeg", f.read())
        mail.send(m)
        r = Message("✅ Thank you!", sender=app.config['MAIL_USERNAME'], recipients=[email])
        r.body = f"Hi {name},\nWe received your message."
        mail.send(r)
        flash("✅ Message sent!", "success")
    except:
        flash("❌ Failed to send", "danger")
    return redirect('/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)