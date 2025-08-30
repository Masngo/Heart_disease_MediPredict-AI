# Heart_disease_MediPredict-AI

Heart_disease_MediPredict-AI Web Application

Overview

This project is an advanced web-based Heart Disease Prediction Application. It allows users to input health parameters and receive a prediction of heart disease probability using a trained Random Forest Machine Learning model.

The application is divided into a backend API (Flask) and a frontend interface (React.js + Tailwind CSS).


---

Features

User-friendly web interface for entering patient data.

Real-time prediction using Random Forest.

Probability score for prediction confidence.

Clean and modern UI built with Tailwind CSS.

Fully deployable on Heroku/Render (backend) and Vercel/Netlify (frontend).



---

Dataset

Heart Disease UCI Dataset (Kaggle)

Features used include: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.



---

Project Structure

heart-disease-webapp/

│

├── backend/

│   ├── app.py

# Flask API

│   ├── utils.py

# Preprocessing functions

│   ├── model.pkl 

# Pre-trained Random Forest model

│   ├── scaler.pkl 

# Pre-fitted scaler

│   ├── requirements.txt     # 

Backend dependencies

│

├── frontend/

│  ├── public/

│   │   └── index.html

│   ├── src/

│   │   ├── App.js

│   │   ├── index.js

│   │   └── components/

│   │       ├── FormInput.js

│   │       └── ResultCard.js

│   └── package.json         # 

Frontend dependencies

│

├── README.md

└── .gitignore


---

Backend Setup

1. Navigate to backend folder:



cd backend

2. Install dependencies:



pip install -r requirements.txt

3. Run Flask API:



python app.py

The API will run on http://127.0.0.1:5000/



---

Frontend Setup

1. Navigate to frontend folder:



cd frontend

2. Install dependencies:



npm install

3. Start React app:



npm start

The frontend will run on http://localhost:3000/

Ensure the fetch URL in App.js points to the backend API.



---

Deployment

Backend: Heroku / Render

Push backend folder to Git repository.

Deploy following Heroku/Render instructions.


Frontend: Vercel / Netlify

Push frontend folder to GitHub.

Import project in Vercel/Netlify.

Update backend API URL in React fetch call.

Deploy.



---

Usage

Open the frontend web page.

Fill in all required patient parameters.

Click Predict.

View prediction (Yes/No) and probability.



---

Notes

The app uses a pre-trained Random Forest model and a pre-fitted scaler.

Ensure backend and frontend are properly connected for live predictions.

This project is portfolio-ready for healthcare AI/ML demonstration.



---

Author

Masango Dewheretsoko

---

License

MIT License



I've created a professional README file for your Heart Disease Prediction Web Application. It includes:

Project overview and features

Dataset information

Full folder structure

Backend and frontend setup instructions

Deployment guide

Usage instructions

Author and license info



