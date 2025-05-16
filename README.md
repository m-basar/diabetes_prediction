# Diabetes Prediction System

## Important Notice
**Note**: This repository only contains the EDA and model training code. The full implementation (web application, API, authentication) is not publicly available. If you're interested in the complete codebase, please contact me directly.

## Overview
A comprehensive web application for diabetes prediction using machine learning. This system provides accurate risk assessment, personalized insights, and detailed explanations for healthcare professionals and patients.

## Features
- **Diabetes Risk Prediction**: Uses machine learning to predict diabetes risk
- **Multi-level User Access**:
  - Admin: System management and oversight
  - Doctor: Patient management and detailed analytics
  - Patient: Personal risk assessment and recommendations
- **Explainable AI**:
  - SHAP (SHapley Additive exPlanations) visualizations
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Permutation importance analysis
- **Robust Backend**:
  - Secure user authentication system
  - Automated visualization cleanup
  - Comprehensive logging

## Installation

```bash
# Clone the repository
git clone https://github.com/m-basar/diabetes_prediction.git
cd diabetes_prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Flask application
python app.py
```

Navigate to `http://localhost:5000` in your web browser.

### Default Users
- Admin: username `admin`, password `admin123`
- Doctor: username `doctor`, password `doctor123`
- Patient: username `patient`, password `patient123`

## Project Structure
```
diabetes_prediction/
│
├── app.py                  # Main Flask application
├── cleanup.py              # Automatic cleanup utilities
├── log.py                  # Logging configuration
├── data/                   # Dataset directory
│   └── Healthcare-Diabetes.csv
├── saved_models/           # Trained machine learning models
├── static/                 # Static files (CSS, JS, images)
│   └── images/
│       └── explanations/   # Generated explanation visualizations
├── templates/              # HTML templates
└── logs/                   # Application logs
```

## Technology Stack
- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, SHAP, LIME
- **Visualization**: Matplotlib, Seaborn
- **Scheduling**: APScheduler
- **Authentication**: Flask-Login

## Requirements
- Python 3.8+
- Dependencies listed in requirements.txt:
  - flask
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - shap
  - lime
  - joblib
  - apscheduler
  - flask-login

## Development
To contribute to this project:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact
For access to the full implementation code or any questions about this project, please reach out to me directly.

## License
MIT License

## Acknowledgements
- The dataset used is based on the [Healthcare-Diabetes dataset from Kaggle](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes)

- This project was created as part of my BSc Computer Science Independent Study

---

**Note**: This system is designed for educational and research purposes and should not replace professional medical advice.
