CIFAR-10 AI Image Generator (WGAN-GP)
A fun, interactive web app that generates brand-new, artificial images!

This project uses an AI model (a WGAN-GP) trained on the CIFAR-10 dataset to create images from scratch. It features a fast FastAPI backend to run the model and an interactive Streamlit frontend designed like an AI "Slot Machine."

Tech Stack
AI Model: TensorFlow / Keras 3

Backend: FastAPI (Python)

Frontend: Streamlit

Dataset: CIFAR-10

Project Structure
Plaintext
WGAN_Project/
├── backend/
│   ├── main.py            # Runs the FastAPI server
│   └── generator.keras    # The trained AI model (add this!)
├── frontend/
│   └── app.py             # Runs the Streamlit dashboard
└── README.md              # Project documentation
Setup & Installation
1. Get the files

Bash
git clone https://github.com/yugj3323j/ATML-Lab
cd WGAN-CIFAR10
2. Install requirements

Bash
pip install fastapi uvicorn tensorflow streamlit pillow requests numpy
3. Add your model
Make sure your trained generator.keras file is placed inside the backend/ folder.

How to Run the App
You will need to run the backend and frontend at the same time. Open two separate terminal windows.

Terminal 1: Start the Backend API

Bash
cd backend
python main.py
Terminal 2: Start the Frontend UI

Bash
cd frontend
streamlit run app.py
Features
AI Slot Machine: "Spin" the GAN! Watch a fun shuffling animation before the AI generates three brand-new 32x32 images.

Real vs. Fake Comparison: Test the AI's skills by comparing a real image from the CIFAR-10 dataset side-by-side with an AI-generated one.

How the AI Works
This project uses a Wasserstein GAN with Gradient Penalty (WGAN-GP). Standard GANs often struggle with "mode collapse" (where the AI gets lazy and generates the exact same image over and over). WGAN-GP fixes this by using advanced math (Earth Mover's Distance) to ensure the AI learns steadily and produces a high variety of clear images.

License
This project is open-source and available under the MIT License.
