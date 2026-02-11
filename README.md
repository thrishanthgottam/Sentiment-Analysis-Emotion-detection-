# Emotion & Sentiment Analysis Web Application

## 📌 Project Overview

This project is a Flask-based web application that performs:

* 📝 **Text Sentiment Analysis** (Positive / Negative)
* 😊 **Real-Time Facial Emotion Detection** using OpenCV and Deep Learning

The application combines Natural Language Processing (NLP) and Computer Vision to analyze both text input and live webcam video.

---

## 🚀 Features

* Text preprocessing with:

  * Stopword removal
  * Stemming (Porter Stemmer)
  * CountVectorizer
* Sentiment prediction using XGBoost model
* Real-time face detection using Haar Cascade
* Emotion classification using a trained CNN model (.h5)
* Live webcam streaming with emotion labels

---

## 🛠 Technologies Used

* Python
* Flask
* OpenCV
* Keras / TensorFlow
* NLTK
* Scikit-learn
* XGBoost
* NumPy

---

## 📂 Project Structure

```
├── app.py
├── model.h5
├── haarcascade_frontalface_default.xml
├── Models/
│   ├── model_xgb.pkl
│   ├── scaler.pkl
│   └── countVectorizer.pkl
├── templates/
│   └── index2.html
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create virtual environment (recommended)

```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Download NLTK stopwords

```
import nltk
nltk.download('stopwords')
```

### 5️⃣ Run the application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🌐 Deployment

This project requires backend hosting such as:

* Render
* Railway
* Heroku

GitHub Pages will NOT work because this is a Flask backend application.

---

## 📊 Models Used

* XGBoost Classifier for Sentiment Analysis
* CNN Model (.h5) for Emotion Detection
* Haar Cascade for Face Detection

---

## 📸 Output

* Text sentiment prediction (Positive / Negative)
* Real-time facial emotion detection:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise
