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


# ── Slot Machine ────────────────────────────────────────────────────────────
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
            "Couldn't fetch images from the backend. "
            "Make sure the FastAPI server is running and the model is loaded."
        )
    else:
        for i, b64 in enumerate(imgs_b64):
            img = Image.open(BytesIO(base64.b64decode(b64)))
            placeholders[i].image(img, caption=f"Result {i+1}", use_container_width=True)

st.markdown("---")

# ── Real vs Generated ────────────────────────────────────────────────────────
st.header("🔍 Real vs Generated")

# Initialise all session-state keys exactly once
for key, default in [
    ("rvsg_ready", False),
    ("rvsg_generated_side", None),
    ("rvsg_left_img_bytes", None),
    ("rvsg_right_img_bytes", None),
    ("rvsg_user_choice", "Left"),   # persists radio selection across reruns
    ("rvsg_checked", False),        # True only after "Check Answer" is clicked
    ("rvsg_error", None),           # surface backend errors clearly
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _start_round() -> None:
    """Load one real CIFAR-10 image and one generated image into session state."""
    st.session_state.rvsg_ready = False
    st.session_state.rvsg_checked = False
    st.session_state.rvsg_user_choice = "Left"
    st.session_state.rvsg_error = None

    imgs_b64 = fetch_images(1)
    if not imgs_b64:
        st.session_state.rvsg_error = (
            "Backend returned no images. "
            "Is the FastAPI server running and the model loaded? "
            f"Try opening: {BACKEND_URL}/debug"
        )
        return

    (x_train, _), _ = cifar10.load_data()
    real = Image.fromarray(x_train[random.randint(0, min(5000, len(x_train) - 1))])
    gen  = Image.open(BytesIO(base64.b64decode(imgs_b64[0]))).convert("RGB")

    buf_real, buf_gen = BytesIO(), BytesIO()
    real.save(buf_real, format="PNG")
    gen.save(buf_gen,  format="PNG")

    if random.random() < 0.5:
        st.session_state.rvsg_left_img_bytes  = buf_real.getvalue()
        st.session_state.rvsg_right_img_bytes = buf_gen.getvalue()
        st.session_state.rvsg_generated_side  = "Right"
    else:
        st.session_state.rvsg_left_img_bytes  = buf_gen.getvalue()
        st.session_state.rvsg_right_img_bytes = buf_real.getvalue()
        st.session_state.rvsg_generated_side  = "Left"

    st.session_state.rvsg_ready = True


# ── Buttons row ──────────────────────────────────────────────────────────────
col_btn, col_hint = st.columns([1, 2])
with col_btn:
    if st.button("Compare (New Round)"):
        _start_round()
with col_hint:
    st.caption("Pick which side you think is generated, then check your answer.")

# ── Error banner ─────────────────────────────────────────────────────────────
if st.session_state.rvsg_error:
    st.error(st.session_state.rvsg_error)

# ── Not ready yet ─────────────────────────────────────────────────────────────
if not st.session_state.rvsg_ready:
    if not st.session_state.rvsg_error:
        st.info("Click **Compare (New Round)** to load one real CIFAR-10 image and one generated image.")
    if st.button("Trouble connecting to backend?"):
        st.write("Make sure the backend is running and open this URL in your browser to test:")
        st.code(f"{BACKEND_URL}/debug")

# ── Game UI ──────────────────────────────────────────────────────────────────
else:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Left")
        st.image(st.session_state.rvsg_left_img_bytes, use_container_width=True)
    with c2:
        st.subheader("Right")
        st.image(st.session_state.rvsg_right_img_bytes, use_container_width=True)

    # KEY FIX: using `key=` binds the radio directly to session_state.
    # Without this, Streamlit resets the widget to index=0 on every rerun
    # (e.g. when "Check Answer" is clicked), losing the user's selection.
    st.radio(
        "Which side is generated?",
        options=["Left", "Right"],
        horizontal=True,
        key="rvsg_user_choice",
    )

    if st.button("Check Answer", type="primary"):
        st.session_state.rvsg_checked = True

    if st.session_state.rvsg_checked:
        correct = st.session_state.rvsg_generated_side
        chosen  = st.session_state.rvsg_user_choice
        if chosen == correct:
            st.success(f"✅ Correct — the **{correct}** image is AI-generated!")
        else:
            st.error(f"❌ Not quite — the **{correct}** image is AI-generated.")
