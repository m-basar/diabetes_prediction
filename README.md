# Diabetes Analyser - Explainable AI System
<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-2.0-black?logo=flask&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

A comprehensive web application for diabetes prediction using machine learning, built as part of my BSc Computer Science degree. This system provides accurate risk assessment, personalised insights, and detailed explanations using **SHAP** and **LIME** to make AI transparent for healthcare professionals and patients.

> ### **Important Notice**
> This public repository contains the core logic (EDA, model training, etc.). The full Flask web application with its user interface and multi-level authentication is in a **private repository**. Please [contact me](#-contact) directly for enquiries about the complete implementation.

---

### 📸 Project Showcase

<table>
  <tr>
    <td align="center"><strong>Login & User Roles</strong></td>
    <td align="center"><strong>Prediction & Results</strong></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/27d29c3b-0e82-437d-a218-3e39b6a3bb1a" alt="Login Screen"></td>
    <td><img src="https://github.com/user-attachments/assets/585849a8-1568-424d-9b13-8488162a8c3b" alt="Prediction Results"></td>
  </tr>
  <tr>
    <td align="center"><strong>SHAP Explanations</strong></td>
    <td align="center"><strong>LIME Explanations</strong></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0d55dd2f-ea5f-4571-824a-3f3a5b5efc7c" alt="SHAP Plot"></td>
    <td><img src="https://github.com/user-attachments/assets/a1fba249-9c37-486d-b507-98b8b79b94c2" alt="LIME Plot"></td>
  </tr>
</table>

---

### 🚀 Core Features

- **Accurate Risk Prediction**: Utilises machine learning to predict diabetes risk with high precision.
- **Explainable AI (XAI)**: Integrates SHAP, LIME, and permutation importance to explain *why* a prediction was made.
- **Multi-level User Access**: Secure roles for Admins, Doctors, and Patients, each with tailored dashboards and permissions.
- **Robust Backend**: Features secure user authentication, automated task scheduling with APScheduler, and comprehensive logging.

---

### 🛠️ Technology Stack

| Category             | Technologies                                            |
| -------------------- | ------------------------------------------------------- |
| **Backend** | `Python`, `Flask`                                       |
| **Data & ML** | `Pandas`, `NumPy`, `Scikit-learn`, `SHAP`, `LIME`       |
| **Authentication** | `Flask-Login`                                           |
| **Scheduling** | `APScheduler`                                           |
| **Visualisation** | `Matplotlib`, `Seaborn`                                 |

---

### ⚙️ Local Setup & Usage

Follow these steps to get the system running locally.

**1. Clone the Repository**
```bash
git clone [https://github.com/m-basar/diabetes_prediction.git](https://github.com/m-basar/diabetes_prediction.git)
cd diabetes_prediction
```

**2. Create and Activate a Virtual Environment**
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Application**
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000` in your browser.

**5. Default Login Credentials**
- **Admin:** `admin` / `admin123`
- **Doctor:** `doctor` / `doctor123`
- **Patient:** `patient` / `patient123`

---

### 📖 Detailed User Guide & Troubleshooting

For a more detailed walkthrough of the system's features, common issues, and error handling, please expand the section below.

<details>
<summary><b>Click to expand User Guide</b></summary>
  
### Using the System

1.  **Login**: Use the provided test credentials for Admin, Doctor, or Patient roles.
2.  **Navigation**: The main dashboard provides a system overview. Use the navigation menu to access features specific to your user role.
3.  **Making Predictions**: Input patient data in the form, click "Predict", and view the detailed risk assessment and explanations.
4.  **Understanding Results**: The system provides probability scores and uses SHAP/LIME to show feature importance. Red indicators suggest factors that increase risk, while green indicators suggest factors that decrease it.
5.  **Viewing Visualisations**: The Exploratory Data Analysis (EDA) section contains interactive plots.

### Troubleshooting

1.  **Common Issues**:
    -   If the application doesn't start, ensure all dependencies from `requirements.txt` are correctly installed in your virtual environment.
    -   Check if port `5000` is available on your system.
    -   Verify you are using Python 3.8 or higher.
2.  **Error Handling**:
    -   Check the `logs` directory for detailed error messages.
    -   Ensure file permissions are correctly set in the project directory.
3.  **Support**:
    -   For technical issues, first consult the logs.
    -   For access issues, contact the system administrator.
    -   For clinical questions, please consult with qualified healthcare professionals.

> **Disclaimer**: This system is designed for educational and research purposes and is not a substitute for professional medical advice.

</details>

---

### 🤝 Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewFeature`).
3.  Commit your changes (`git commit -m 'Add some NewFeature'`).
4.  Push to the branch (`git push origin feature/NewFeature`).
5.  Open a Pull Request.

---

### 📜 License & Acknowledgements

- This project is licensed under the **MIT License**.
- The dataset used is based on the [Healthcare-Diabetes dataset from Kaggle](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes).

---

### 📞 Contact

If you have any questions or would like to discuss the full implementation, please don't hesitate to reach out.
