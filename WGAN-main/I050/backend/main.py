import base64
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent / "generator.keras"

generator = None
latent_dim = 100
try:
    generator = tf.keras.models.load_model(MODEL_PATH)
    generator.trainable = False
    try:
        inferred = getattr(generator, "input_shape", None)
        if isinstance(inferred, (list, tuple)) and len(inferred) >= 2 and inferred[-1] is not None:
            latent_dim = int(inferred[-1])
    except Exception:
        pass
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")


@app.get("/generate")
async def generate(n: int = 1):
    if n < 1 or n > 64:
        raise HTTPException(status_code=400, detail="n must be between 1 and 64")
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Place your generator at {MODEL_PATH}",
        )

    noise = tf.random.normal(shape=(n, latent_dim))
    generated_imgs = generator.predict(noise, verbose=0)

    generated_imgs = generated_imgs.numpy() if tf.is_tensor(generated_imgs) else generated_imgs
    generated_imgs = np.asarray(generated_imgs)

    if generated_imgs.ndim != 4:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected generator output shape: {generated_imgs.shape}",
        )

    # Handle common channel shapes
    if generated_imgs.shape[-1] == 1:
        generated_imgs = np.repeat(generated_imgs, 3, axis=-1)
    elif generated_imgs.shape[-1] != 3:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected channel count: {generated_imgs.shape[-1]}",
        )

    # Auto-detect output range and scale to [0, 255]
    out_min = float(np.nanmin(generated_imgs))
    out_max = float(np.nanmax(generated_imgs))
    if not np.isfinite(out_min) or not np.isfinite(out_max):
        raise HTTPException(status_code=500, detail="Generator output contains NaN/Inf")

    if 0.0 <= out_min and out_max <= 1.0:
        generated_imgs = generated_imgs * 255.0
    elif -1.0 <= out_min and out_max <= 1.0:
        generated_imgs = generated_imgs * 127.5 + 127.5
    # else: assume already in roughly [0, 255]

    generated_imgs = np.clip(generated_imgs, 0, 255).astype(np.uint8)

    b64_images: list[str] = []
    for i in range(n):
        img = Image.fromarray(generated_imgs[i])
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        b64_images.append(img_str)

    return {"images": b64_images}


@app.get("/debug")
async def debug():
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Place your generator at {MODEL_PATH}",
        )

    noise = tf.random.normal(shape=(1, latent_dim))
    out = generator.predict(noise, verbose=0)
    out = out.numpy() if tf.is_tensor(out) else out
    out = np.asarray(out)
    return {
        "model_path": str(MODEL_PATH),
        "latent_dim": latent_dim,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_min": float(np.nanmin(out)),
        "output_max": float(np.nanmax(out)),
        "output_mean": float(np.nanmean(out)),
    }
