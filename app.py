import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "model.h5"  # ← apne dost ka model file naam
IMG_SIZE    = 64           # ← agar model alag size pe trained hai toh change karo
HOLD_SECS   = 1.5

CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("🤟 ASL Sign Language Translator")

@st.cache(allow_output_mutation=True)
def load_asl_model():
    return load_model(MODEL_PATH)

try:
    model = load_asl_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model load nahi hua: {e}\nModel.h5 file WEBCAM folder mein rakho.")
    model_loaded = False

if "sentence"      not in st.session_state: st.session_state.sentence      = ""
if "last_letter"   not in st.session_state: st.session_state.last_letter   = ""
if "letter_start"  not in st.session_state: st.session_state.letter_start  = None
if "letter_locked" not in st.session_state: st.session_state.letter_locked = False

def predict(frame_bgr):
    img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASSES[idx], float(preds[idx])

if model_loaded:
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Current Letter")
        letter_box = st.empty()
        letter_box.markdown("## —")

        st.subheader("Sentence")
        sentence_box = st.empty()
        sentence_box.markdown(f"### `{st.session_state.sentence or '...'}`")

        if st.button("Delete Last"):
            st.session_state.sentence = st.session_state.sentence[:-1]
        if st.button("Clear All"):
            st.session_state.sentence = ""

        st.markdown("---")
        st.markdown("""
        **How to use:**
        - Hold sign steady for **1.5 sec** → letter adds
        - `nothing` = reset/pause
        - `del` = delete last letter
        - `space` = add space
        """)

    with col1:
        run = st.checkbox("Start Camera")
        frame_box = st.empty()

        if run:
            cap = cv2.VideoCapture(0)

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera nahi mili!")
                    break

                frame = cv2.flip(frame, 1)
                label, conf = predict(frame)

                now = time.time()
                if label == st.session_state.last_letter:
                    if st.session_state.letter_start is None:
                        st.session_state.letter_start = now
                    elapsed  = now - st.session_state.letter_start
                    progress = min(elapsed / HOLD_SECS, 1.0)

                    if elapsed >= HOLD_SECS and not st.session_state.letter_locked:
                        st.session_state.letter_locked = True
                        if label == "del":
                            st.session_state.sentence = st.session_state.sentence[:-1]
                        elif label == "space":
                            st.session_state.sentence += " "
                        elif label != "nothing":
                            st.session_state.sentence += label
                else:
                    st.session_state.last_letter   = label
                    st.session_state.letter_start  = now
                    st.session_state.letter_locked = False
                    progress = 0.0

                h, w = frame.shape[:2]
                bar_w = int(w * progress)
                cv2.rectangle(frame, (0, h-10), (bar_w, h), (0, 255, 100), -1)
                cv2.rectangle(frame, (0, h-10), (w, h),     (50, 50, 50),  1)

                lock = " LOCKED" if st.session_state.letter_locked else ""
                disp = label if label != "nothing" else "."
                letter_box.markdown(f"## {disp}{lock}  ({conf*100:.0f}%)")
                sentence_box.markdown(f"### `{st.session_state.sentence or '...'}`")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_box.image(frame_rgb, channels="RGB", width=600)

            cap.release()