import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, Response
from flask_cors import CORS
from models import GiMeFive
import dlib

# ---------------- CONFIG ----------------
app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# โหลดโมเดล
model_path = "./models/best_GiMeFive.pth"
model = GiMeFive().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------- Grad-CAM Hook ----------------
class Hook():
    def __init__(self):
        self.hook_forward = None
        self.hook_backward = None
        self.forward_out = None
        self.backward_out = None

    def hook_fn_forward(self, module, input, output):
        self.forward_out = output

    def hook_fn_backward(self, module, grad_input, grad_output):
        self.backward_out = grad_output[0] 

    def register_hook(self, module):
        self.hook_forward = module.register_forward_hook(self.hook_fn_forward)
        self.hook_backward = module.register_full_backward_hook(self.hook_fn_backward)

    def unregister_hook(self):
        self.hook_forward.remove()
        self.hook_backward.remove()

# hook Grad-CAM ที่ conv5
final_layer = model.conv5
hook = Hook()
hook.register_hook(final_layer)

# ---------- Hook สำหรับ Layer Activation Visualization ----------
class ActivationHook():
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.save_output)

    def save_output(self, module, input, output):
        self.features = output.detach().cpu()

    def remove(self):
        self.hook.remove()

# activation layer (ปรับได้: conv1 / conv2 / conv3 / conv4)
activation_layer = model.conv3
activation_hook = ActivationHook(activation_layer)

# class labels
class_labels = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# dlib predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# display settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (154, 1, 254)
thickness = 2
line_type = cv2.LINE_AA
transparency = 0.4
evaluation_frequency = 5

# ---------- สร้างภาพ Activation Visualization ----------
def visualize_activations(activations, size=64, num_maps=6):
    activations = activations[0]  # (C, H, W)
    activations = activations[:num_maps]  # เอาเฉพาะบางแผนที่

    grid_images = []

    for i in range(num_maps):
        act = activations[i].numpy()

        act = act - act.min()
        if act.max() != 0:
            act = act / act.max()
        act = (act * 255).astype(np.uint8)

        act_resized = cv2.resize(act, (size, size))
        act_resized = cv2.applyColorMap(act_resized, cv2.COLORMAP_JET)
        grid_images.append(act_resized)

    grid = np.hstack(grid_images)
    return grid

# ---------- คำนวณ Grad-CAM ----------
def detect_emotion_with_heatmap(face_img):
    image = Image.fromarray(face_img)
    image_tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    scores = probs.detach().cpu().numpy().flatten()
    max_index = np.argmax(scores)

    # backward สำหรับ Grad-CAM
    one_hot = torch.zeros_like(outputs)
    one_hot[0, max_index] = 1
    outputs.backward(gradient=one_hot, retain_graph=True)

    gradients = hook.backward_out
    activations = hook.forward_out

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam.cpu().detach().numpy()

    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) != 0 else np.zeros_like(cam)

    return scores, cam

# ---------- overlay heatmap ----------
def overlay_heatmap_on_face(frame, x, y, w, h, cam):
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    roi = frame[y:y+h, x:x+w, :]
    overlay = heatmap * transparency + roi / 255 * (1 - transparency)
    frame[y:y+h, x:x+w, :] = np.uint8(255 * overlay)

# ---------- Streaming ----------
def generate_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    max_emotion = ''
    scores = [0] * len(class_labels)
    cam = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            crop_img = frame[y:y+h, x:x+w]

            if counter == 0 or cam is None:
                scores, cam = detect_emotion_with_heatmap(crop_img)
                max_emotion = class_labels[np.argmax(scores)]

            if cam is not None:
                overlay_heatmap_on_face(frame, x, y, w, h, cam)

            cv2.putText(frame, max_emotion, (x, y-10),
                        font, font_scale, font_color, thickness, line_type)

            # แสดงคะแนน emotion
            for idx, label in enumerate(class_labels):
                text = f"{label}: {scores[idx]:.2f}"
                cv2.putText(frame, text, (x+w+10, y + idx*25),
                            font, 0.6, (0,255,0), 1, line_type)

            # DLIB landmarks
            dlib_rect = dlib.rectangle(x, y, x+w, y+h)
            landmarks = predictor(gray, dlib_rect)
            for i in range(68):
                lx = landmarks.part(i).x
                ly = landmarks.part(i).y
                cv2.circle(frame, (lx, ly), 1, (0,0,255), -1)

            # ---------- เพิ่ม Layer Activation Visualization ----------
            if activation_hook.features is not None:
                act_grid = visualize_activations(activation_hook.features, size=50, num_maps=6)
                h2, w2, _ = act_grid.shape
                # คำนวณตำแหน่งให้อยู่กลางล่าง
                x_start = frame.shape[1] // 2 - w2 // 2  # กลางแนวนอน
                y_start = frame.shape[0] - h2           # ล่างสุด

                frame[y_start:frame.shape[0], x_start:x_start + w2] = act_grid

        counter += 1
        if counter >= evaluation_frequency:
            counter = 0

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
