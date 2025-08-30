# Heart_disease_MediPredict-AI

Project Overview
The Heart Disease MediPredict-AI App is a full-stack web application designed to predict the presence of heart disease. It serves as a practical demonstration of how a machine learning model, initially developed in a data science notebook, can be integrated into a functional, user-friendly web interface.

The application allows users to input key patient data through a simple form. This data is then sent to a Python backend, which uses a pre-trained Random Forest Classifier to generate a real-time prediction.

Key Features
Intuitive Interface: A clean, responsive, single-page web form for data entry.

Real-time Prediction: Uses a machine learning model to provide instant results.

Full-Stack Architecture: Combines a modern frontend with a robust Python backend.

Scalable: Structured for easy deployment on cloud platforms.

Tech Stack
Frontend: HTML, JavaScript, Tailwind CSS

Backend: Python, Flask, Gunicorn (for production)

Machine Learning: Scikit-learn, pandas, joblib

Project Structure
The repository is organized to follow best practices for a Flask application.

Heart_disease_MediPredict-AI/

├── app.py

├── heart_disease_model.pkl

├── Procfile

├── README.md

├── requirements.txt

├── scaler.pkl

└── templates/
    └── index.html

Getting Started (Local Development)
To run this project on your local machine, follow these steps:

Clone the Repository:

git clone [https://github.com/Masango/Heart_disease_MediPredict-AI.git](https://github.com/Masango/Heart_disease_MediPredict-AI.git)
cd Heart_disease_MediPredict-AI

Create and Activate a Virtual Environment:

python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Run the Application:

python app.py

The application will be accessible at http://127.0.0.1:5000 in your web browser.

Deployment
This project is configured for seamless deployment on platforms like Render. The included Procfile and requirements.txt files ensure that the application can be built and run automatically in a production environment.

Contact
For questions or feedback, feel free to connect with me:

LinkedIn Profile: [linkedin.com/in/masango-dewheretsoko-5ba182148 ](https://www.linkedin.com/in/masango-dewheretsoko-5ba182148)
