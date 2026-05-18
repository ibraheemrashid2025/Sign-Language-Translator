import os
import time
import numpy as np
import streamlit as st
import tflite_runtime.interpreter as tflite
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from PIL import Image, ImageDraw

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "dataset/alphabet_cnn_model.tflite"  
IMG_SIZE    = 64           
HOLD_SECS   = 1.5  

CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("🤟 ASL Sign Language Translator")

# File & Size Checker
if os.path.exists(MODEL_PATH):
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    st.info(f"📁 Model File Found! Size: **{file_size_mb:.2f} MB**")
else:
    st.error(f"❌ Error: Model file '{MODEL_PATH}' GitHub par nahi mili!")
    st.stop()

# Safe Global Loading
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
    st.error(f"❌ TFLite Interpreter file ko read nahi kar pa raha: {interpreter_obj}")
    st.stop()
else:
    st.success("✅ AI Model Perfectly Loaded & Ready!")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.interpreter = interpreter_obj
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Realtime States
        self.sentence = ""
        self.last_letter = ""
        self.letter_start = None
        self.letter_locked = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_rgb = frame.to_ndarray(format="rgb24")
        img_rgb = img_rgb[:, ::-1, :]  # Mirror effect
        h, w, _ = img_rgb.shape
        
        try:
            pil_img = Image.fromarray(img_rgb)
            pil_img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
            input_data = np.array(pil_img_resized).astype("float32") / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            
            # Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            idx = np.argmax(output_data)
            label = CLASSES[idx]
            conf = float(output_data[idx])
            
            draw = ImageDraw.Draw(pil_img)
            now = time.time()
            progress = 0.0
            
            if conf > 0.70:
                if label == self.last_letter:
                    if self.letter_start is None:
                        self.letter_start = now
                    elapsed = now - self.letter_start
                    progress = min(elapsed / HOLD_SECS, 1.0)
                    
                    if elapsed >= HOLD_SECS and not self.letter_locked:
                        self.letter_locked = True
                        if label == "del":
                            self.sentence = self.sentence[:-1]
                        elif label == "space":
                            self.sentence += " "
                        elif label != "nothing":
                            self.sentence += label
                else:
                    self.last_letter = label
                    self.letter_start = now
                    self.letter_locked = False
            else:
                progress = 0.0
                
            # Live Drawing using PIL
            bar_w = int(w * progress)
            draw.rectangle([(0, h - 15), (bar_w, h)], fill=(0, 255, 100))
            draw.rectangle([(0, h - 15), (w, h)], outline=(50, 50, 50), width=1)
            
            lock_status = " [LOCKED]" if self.letter_locked else ""
            disp_label = label if label != "nothing" else "Scanning..."
            
            draw.rectangle([(0, 0), (w, 80)], fill=(0, 0, 0, 128))
            draw.text((20, 10), f"Sign: {disp_label}{lock_status} ({conf*100:.0f}%)", fill=(255, 255, 255))
            draw.text((20, 45), f"Sentence: > {self.sentence if self.sentence else '...'} <", fill=(0, 255, 100))
            
            img_rgb = np.array(pil_img)
                                
        except Exception as e:
            pass
            
        return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")

# UI Layout
st.markdown("---")
st.write("👇 Neechay **Start** button par click karein aur live testing shuru karein!")

webrtc_streamer(
    key="asl-sign-language-translator",
    video_processor_factory=ASLVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
