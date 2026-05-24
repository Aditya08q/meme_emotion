import cv2
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
from PIL import Image, ImageSequence
import json


MODEL_PATH = "models/emotion_model.h5"     
MEME_DIR = "meme"
IMG_SIZE = (48,48)
SCORE_THRESHOLD = 0.35

model = load_model(MODEL_PATH)

detector = MTCNN()

def build_meme_map(meme_dir):
    meme_map = {}
    for emo in os.listdir(meme_dir):
        full = os.path.join(meme_dir, emo)
        if os.path.isdir(full):
            meme_map[emo] = [os.path.join(full,f) for f in os.listdir(full) 
                             if f.lower().endswith(('.png','.jpg','.jpeg','.gif','.mp4'))]
    return meme_map

meme_map = build_meme_map(MEME_DIR)
print("Loaded memes for:", {k: len(v) for k,v in meme_map.items()})

with open("models/label_map.json") as f:
    label_map = json.load(f)

EMOTIONS = [label for label, idx in sorted(label_map.items(), key=lambda x: x[1])]
print("Loaded emotion order:", EMOTIONS)

def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    arr = resized.astype('float32')/255.0
    arr = np.expand_dims(arr, -1)   # (h,w,1)
    arr = np.expand_dims(arr, 0)    # (1,h,w,1)
    return arr

def pick_meme(emotion):
    choices = meme_map.get(emotion, [])
    return random.choice(choices) if choices else None

def overlay_image(frame, img_path, x=0, y=0, max_w=320, max_h=320):
    img = Image.open(img_path).convert("RGBA")
    w,h = img.size
    scale = min(max_w/w, max_h/h, 1.0)
    img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    frame_pil.paste(img, (x,y), img)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)

def overlay_gif(frame, gif_path, x=0, y=0, max_w=320, max_h=320, frame_idx=0):
    gif = Image.open(gif_path)
    frames = [f.copy().convert("RGBA") for f in ImageSequence.Iterator(gif)]
    current_frame = frames[frame_idx % len(frames)]
    w,h = current_frame.size
    scale = min(max_w/w, max_h/h, 1.0)
    current_frame = current_frame.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    frame_pil.paste(current_frame, (x,y), current_frame)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)

class VideoOverlay:
    def __init__(self, path, max_w=320, max_h=320):
        self.cap = cv2.VideoCapture(path)
        self.max_w = max_w
        self.max_h = max_h
    
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if ret:
            h,w,_ = frame.shape
            scale = min(self.max_w/w, self.max_h/h, 1.0)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        return frame

def get_meme_overlay(obj):
    ext = os.path.splitext(obj)[1].lower()
    if ext in ['.png','.jpg','.jpeg']:
        return 'image'
    elif ext == '.gif':
        return 'gif'
    elif ext in ['.mp4','.avi']:
        return 'video'
    return 'image'

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

current_emo = None
current_meme = None
frame_count = 0
video_obj = None

while True:
    ret, frame = cap.read()
    if not ret: break

    small = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))
    faces = detector.detect_faces(small)

    for f in faces:
        x,y,w,h = f['box']
        x,y = max(0,x), max(0,y)
        pad = int(0.15*max(w,h))
        x1,y1 = max(0,x-pad), max(0,y-pad)
        x2,y2 = min(small.shape[1], x+w+pad), min(small.shape[0], y+h+pad)
        crop = small[y1:y2, x1:x2]
        if crop.size == 0: continue

        inp = preprocess_face(crop)
        preds = model.predict(inp)
        idx = int(np.argmax(preds))
        prob = float(np.max(preds))
        emo = EMOTIONS[idx] if prob >= SCORE_THRESHOLD else 'neutral'

        if emo != current_emo:
            current_emo = emo
            current_meme = pick_meme(emo)
            frame_count = 0
            video_obj = None
            if current_meme and get_meme_overlay(current_meme) == 'video':
                video_obj = VideoOverlay(current_meme)

        cv2.rectangle(small, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(small, f"{emo} {prob:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    max_w, max_h = 320,320
    x = small.shape[1] - max_w - 10
    y = 10

    if current_meme:
        ext_type = get_meme_overlay(current_meme)
        if ext_type == 'image':
            out = overlay_image(small, current_meme, x=x, y=y, max_w=max_w, max_h=max_h)
        elif ext_type == 'gif':
            out = overlay_gif(small, current_meme, x=x, y=y, max_w=max_w, max_h=max_h, frame_idx=frame_count)
        elif ext_type == 'video' and video_obj:
            vid_frame = video_obj.read_frame()
            if vid_frame is not None:
                frame_pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).convert("RGBA")
                vid_pil = Image.fromarray(cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
                frame_pil.paste(vid_pil, (x,y))
                out = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        frame_count += 1
    else:
        out = small

    cv2.imshow("Emotion -> Meme", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('n'):  # next meme
        current_meme = pick_meme(current_emo)
        frame_count = 0
        video_obj = None
        if current_meme and get_meme_overlay(current_meme) == 'video':
            video_obj = VideoOverlay(current_meme)

cap.release()
cv2.destroyAllWindows()
