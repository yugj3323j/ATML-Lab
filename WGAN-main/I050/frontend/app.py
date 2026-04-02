import base64
import random
import time
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image
from tensorflow.keras.datasets import cifar10

st.set_page_config(page_title="WGAN CIFAR-10", layout="wide")

BACKEND_URL = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000").rstrip("/")

st.title("🎨 WGAN-GP Dashboard")


def fetch_images(n: int = 1) -> list[str]:
    try:
        res = requests.get(f"{BACKEND_URL}/generate", params={"n": n}, timeout=10)
        if res.status_code != 200:
            return []
        payload = res.json()
        images = payload.get("images", [])
        return images if isinstance(images, list) else []
    except Exception:
        return []


st.header("🎰 AI Slot Machine")
st.caption("Click to generate images; a quick animation plays while we wait.")

if st.button("Spin the GAN", type="primary"):
    cols = st.columns(3)
    placeholders = [col.empty() for col in cols]

    for _ in range(8):
        for ph in placeholders:
            ph.image(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
                use_container_width=True,
            )
        time.sleep(0.1)

    imgs_b64 = fetch_images(3)
    if len(imgs_b64) != 3:
        st.error(
            "Couldn’t fetch images from the backend. "
            "Make sure the FastAPI server is running and the model is loaded."
        )
    else:
        for i, b64 in enumerate(imgs_b64):
            img = Image.open(BytesIO(base64.b64decode(b64)))
            placeholders[i].image(img, caption=f"Result {i+1}", use_container_width=True)

st.markdown("---")
st.header("🔍 Real vs Generated")

if "rvsg_ready" not in st.session_state:
    st.session_state.rvsg_ready = False
if "rvsg_generated_side" not in st.session_state:
    st.session_state.rvsg_generated_side = None
if "rvsg_left_img_bytes" not in st.session_state:
    st.session_state.rvsg_left_img_bytes = None
if "rvsg_right_img_bytes" not in st.session_state:
    st.session_state.rvsg_right_img_bytes = None


def _start_round() -> None:
    (x_train, _), _ = cifar10.load_data()
    real = Image.fromarray(x_train[random.randint(0, min(5000, len(x_train) - 1))])

    imgs_b64 = fetch_images(1)
    if not imgs_b64:
        st.session_state.rvsg_ready = False
        st.session_state.rvsg_generated_side = None
        st.session_state.rvsg_left_img_bytes = None
        st.session_state.rvsg_right_img_bytes = None
        return

    gen = Image.open(BytesIO(base64.b64decode(imgs_b64[0]))).convert("RGB")

    buf_real = BytesIO()
    real.save(buf_real, format="PNG")
    buf_gen = BytesIO()
    gen.save(buf_gen, format="PNG")

    if random.random() < 0.5:
        st.session_state.rvsg_left_img_bytes = buf_real.getvalue()
        st.session_state.rvsg_right_img_bytes = buf_gen.getvalue()
        st.session_state.rvsg_generated_side = "Right"
    else:
        st.session_state.rvsg_left_img_bytes = buf_gen.getvalue()
        st.session_state.rvsg_right_img_bytes = buf_real.getvalue()
        st.session_state.rvsg_generated_side = "Left"

    st.session_state.rvsg_ready = True


col_btn, col_hint = st.columns([1, 2])
with col_btn:
    if st.button("Compare (New Round)"):
        _start_round()
with col_hint:
    st.caption("Pick which side you think is generated, then check your answer.")

if not st.session_state.rvsg_ready:
    st.info("Click **Compare (New Round)** to load one real CIFAR-10 image and one generated image.")
    if st.button("Trouble connecting to backend?"):
        st.write("Make sure the backend is running and try loading:", f"{BACKEND_URL}/debug")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Left")
        st.image(st.session_state.rvsg_left_img_bytes, use_container_width=True)
    with c2:
        st.subheader("Right")
        st.image(st.session_state.rvsg_right_img_bytes, use_container_width=True)

    choice = st.radio(
        "Which side is generated?",
        options=["Left", "Right"],
        horizontal=True,
        index=0,
    )

    if st.button("Check Answer", type="primary"):
        correct_side = st.session_state.rvsg_generated_side
        if choice == correct_side:
            st.success(f"Correct — the **{correct_side}** image is generated.")
        else:
            st.error(f"Not quite — the **{correct_side}** image is generated.")
