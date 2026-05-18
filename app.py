import os
import time
import numpy as np
import streamlit as st
import tflite_runtime.interpreter as tflite
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from PIL import Image, ImageDraw

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "dataset/alphabet_cnn_model.tflite"  
IMG_SIZE    = 64           
HOLD_SECONDS = 1.5  

CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("🤟 ASL Sign Language Translator (MediaPipe Cloud Live)")

# File Check
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' GitHub par nahi mili!")
    st.stop()

# Safe Global Model Loading
@st.cache_resource
def load_my_tflite_model():
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter, True
    except Exception as e:
        return str(e), False

interpreter_obj, load_success = load_my_tflite_model()

if not load_success:
    st.error(f"❌ TFLite Interpreter load nahi ho saka: {interpreter_obj}")
    st.stop()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Main Multi-threaded MediaPipe + TFLite Processor Class
class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.interpreter = interpreter_obj
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # MediaPipe Hands Setup inside thread instance
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        # Aap ki exact local state variables
        self.sentence = ""
        self.last_detected_class = None
        self.hold_start_time = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_rgb = frame.to_ndarray(format="rgb24")
        img_rgb = img_rgb[:, ::-1, :]  # Mirror effect
        h, w, _ = img_rgb.shape
        
        # MediaPipe Processing
        results = self.hands.process(img_rgb)
        
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        predicted_class = "nothing"
        confidence = 0.0
        now = time.time()
        progress = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in hand_landmarks.landmark:
                    x_pixel, y_pixel = int(lm.x * w), int(lm.y * h)
                    if x_pixel < x_min: x_min = x_pixel
                    if y_pixel < y_min: y_min = y_pixel
                    if x_pixel > x_max: x_max = x_pixel
                    if y_pixel > y_max: y_max = y_pixel
                
                # ─── AAP KI EXACT PERFECT SQUARE CROPPING LOGIC ───
                box_w = x_max - x_min
                box_h = y_max - y_min
                size = max(box_w, box_h) + 40  # 40px padding
                
                center_x = x_min + box_w // 2
                center_y = y_min + box_h // 2
                
                x_min = max(0, center_x - size // 2)
                y_min = max(0, center_y - size // 2)
                x_max = min(w, x_min + size)
                y_max = min(h, y_min + size)
                
                # Draw Red Bounding Box on Frame
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(255, 0, 0), width=2)
                
                # Extract Hand Crop Region safely
                if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                    hand_crop = img_rgb[y_min:y_max, x_min:x_max]
                    
                    # Convert to PIL for stable resize without cv2 graphics crash
                    pil_crop = Image.fromarray(hand_crop).resize((IMG_SIZE, IMG_SIZE))
                    
                    # TFLite Preprocessing
                    input_data = np.array(pil_crop).astype("float32") / 255.0
                    input_data = np.expand_dims(input_data, axis=0)
                    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
                    
                    # Run Model
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()
                    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                    
                    confidence = float(np.max(output_data))
                    predicted_class = CLASSES[np.argmax(output_data)]
                    
                    # Draw Current Class atop bounding box
                    if confidence > 0.82:
                        draw.text((x_min, max(0, y_min - 20)), f"{predicted_class} ({confidence*100:.1f}%)", fill=(0, 255, 0))
                        
                        # ─── AAP KI HOLD-TO-COMMIT TIMING LOGIC ───
                        if predicted_class == self.last_detected_class:
                            if self.hold_start_time is None:
                                self.hold_start_time = now
                            
                            elapsed = now - self.hold_start_time
                            progress = min(elapsed / HOLD_SECONDS, 1.0)
                            
                            if elapsed >= HOLD_SECONDS:
                                if predicted_class == "del":
                                    self.sentence = self.sentence[:-1]
                                elif predicted_class == "space":
                                    self.sentence += " "
                                elif predicted_class != "nothing":
                                    self.sentence += predicted_class
                                
                                # Reset for next sign
                                self.hold_start_time = None
                                self.last_detected_class = None
                        else:
                            self.last_detected_class = predicted_class
                            self.hold_start_time = now
                    else:
                        progress = 0.0

        # ─── LIVE INTERFACE OVERLAY DRAWING ───────────────────────────────────
        # 1. Progress Loading Bar at bottom
        bar_w = int(w * progress)
        draw.rectangle([(0, h - 15), (bar_w, h)], fill=(0, 255, 100))
        draw.rectangle([(0, h - 15), (w, h)], outline=(50, 50, 50), width=1)
        
        # 2. Black Header Bar for sentence output
        draw.rectangle([(0, 0), (w, 85)], fill=(0, 0, 0, 180))
        draw.text((20, 15), "SENTENCE:", fill=(0, 255, 0))
        draw.text((20, 45), self.sentence if self.sentence else "[Empty - Hold sign for 1.5s]", fill=(255, 255, 255))
        
        img_final = np.array(pil_img)
        return av.VideoFrame.from_ndarray(img_final, format="rgb24")

# Streamlit Front-End Interface
st.markdown("---")
st.write("👇 Neechay **Start** button dabaayein. MediaPipe haath track karega aur text generate karega:")

webrtc_streamer(
    key="asl-sign-language-translator",
    video_processor_factory=ASLVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
