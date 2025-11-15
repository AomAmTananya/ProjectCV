## Requirements
A detailed list of requirements can be found in requirements.txt. 
```
pip install -r requirements.txt
python3 -m pip install -r requirements.txt
```
## Setup

- PYTHON ENVIRONMENT
```
# สร้าง virtual environment
python -m venv .venv
# เปิดใช้งาน
.\.venv\Scripts\activate      # WINDOWS
source .venv/bin/activate     # MAC/LINUX
# ติดตั้ง pip packages
pip install -r requirements.txt
```
- Node.js packages
```
npm install
```

## Usage
- Run Node.js server
```
node app.js
```
- Run Flask server

Camera Grad - Cam 
```
python Gradcam.py
```
Camera general
```
python predict.py
```

## PROJECT STRUCTURE
ProjectAiapp/
│
├─ models/
│   └─ best_gimefive.pth       # PyTorch model
│
├─ static/
│   └─ heatmaps/               # Stores generated heatmaps
│
├─ uploads/                    # Temporary uploaded images
│
├─ public/
│   └─ css                     # Css
│   └─ assets                  # Photo
│   └─ html              
│ 
├─ predict_torch.py            # Python script for prediction Image
├─ Gradcam.py                  # Python script for prediction + Grad-CAM
├─ predict.py                  # Python script for prediction with general Cam
├─ models.py                   # Models class
├─ app.js                      # Node.js server
└─ requirements.txt            # Python packages
