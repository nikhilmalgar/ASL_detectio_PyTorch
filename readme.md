# 🤟 ASL Hand Sign Recognition using PyTorch

This project is a real-time American Sign Language (ASL) hand sign recognition system built using **PyTorch**, **OpenCV**, and **Streamlit**. It supports webcam input and gives voice feedback for the predicted signs. The model is trained on the **ASL Alphabet Dataset** and integrated into a lightweight, user-friendly app.

---

## 🚀 Features

- 🔍 Real-time ASL hand sign detection via webcam
- 🧠 PyTorch-based CNN model
- 📦 Streamlit UI for live interaction
- 🔊 Voice output using `pyttsx3`
- 📈 Training & evaluation scripts included
- 📊 Accuracy reporting and visualization
- 💾 Save & load trained model checkpoints

---

## 📂 Project Structure

ASL_HANDSIGN_PYTORCH/
│
├── app/ # Streamlit app with real-time webcam prediction
│ └── app.py
│
├── src/ # Source files for the model and dataloader
│ ├── dataset.py
│ ├── evaluate.py
│ └── model.py
│
├── data/ # Dataset and model checkpoints
│ ├── asl_alphabet_train/
│ ├── asl_alphabet_test/
│ └── saved_models/
│
├── train.py # Training script
├── test_dataloader.py # Data loading test script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Files to ignore in Git

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ASL_HANDSIGN_PYTORCH.git
cd ASL_HANDSIGN_PYTORCH

## 2. Create and activate a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

#3. Install dependencies
pip install -r requirements.txt

📁 Dataset Setup
Download the ASL Alphabet Dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/grassknoted/asl-alphabet

After downloading and extracting:

Place training data inside:

bash
Copy
Edit
data/asl_alphabet_train/
Place test data inside:

bash
Copy
Edit
data/asl_alphabet_test/
⚠️ The dataset is not included in the repo due to size.

🧠 Model Training
To train the model on your local machine:

bash
Copy
Edit
python train.py
After training, the model will be saved at:

bash
Copy
Edit
data/saved_models/asl_model.pth
🧪 Test DataLoader
Check if dataset is loading correctly:

bash
Copy
Edit
python test_dataloader.py
🎯 Evaluate the Model
Use evaluate.py to get test accuracy:

bash
Copy
Edit
python src/evaluate.py
🌐 Run the Real-Time Streamlit App
Launch the web app:

bash
Copy
Edit
streamlit run app/app.py
The webcam will turn on. Show ASL signs one at a time to get predictions with audio feedback.

🔊 Voice Output
Voice output is powered by pyttsx3.

For Linux, install this additional package if needed:

bash
Copy
Edit
sudo apt-get install espeak
📝 .gitignore Example
Here’s a sample .gitignore for your repo:

bash
Copy
Edit
__pycache__/
*.pyc
*.pth
*.log
venv/
*.DS_Store
data/saved_models/
data/asl_alphabet_train/
data/asl_alphabet_test/
💡 Tips
Make sure your hand is clearly visible to the webcam

Use plain backgrounds and good lighting

Use same size/image orientation as dataset samples

Train for more epochs or augment the dataset to improve accuracy

📃 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Nikhil Malagar
B.Tech in CSE (AI & ML)
Passionate about AI, Deep Learning, and Full-Stack Development
🔗 GitHub: @nikhilmalgar

⭐ Support
If you find this project useful, please consider giving it a ⭐ on GitHub. Thank you! 😊


```
