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

Note: This repository contains only the training components of the project. For the full implementation including the web application, please contact the repository owner.

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

## User Guide

### System Setup
1. **Prerequisites**:
   - Python 3.8 or higher installed on your system
   - Git installed (optional, for cloning the repository)
   - A modern web browser

2. **Getting Started**:
   ```bash
   # Clone or download the repository
   git clone [repository-url] # Or download and extract the ZIP file
   cd diabetes_prediction

   # Create and activate a virtual environment
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Running the Application**:
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

### Using the System

1. **Login**:
   - Use the provided test credentials:
     - Admin: username `admin`, password `admin123`
     - Doctor: username `doctor`, password `doctor123`
     - Patient: username `patient`, password `patient123`

2. **Navigation**:
   - The dashboard shows an overview of the system
   - Use the navigation menu to access different features
   - Each user role has different permissions and views

3. **Making Predictions**:
   - Input patient data in the provided form
   - Click "Predict" to get the diabetes risk assessment
   - View detailed explanations of the prediction
   - Export results if needed

4. **Viewing Visualizations**:
   - Access the EDA (Exploratory Data Analysis) section
   - Interactive plots are available in the `/diabetes_eda_plots` directory
   - Hover over plots for detailed information

5. **Understanding Results**:
   - The system provides probability scores
   - SHAP and LIME explanations show feature importance
   - Red indicators suggest higher risk factors
   - Green indicators suggest lower risk factors

### Troubleshooting

1. **Common Issues**:
   - If the application doesn't start, ensure all dependencies are installed
   - Check if the required ports are available (default: 5000)
   - Verify Python version compatibility

2. **Error Handling**:
   - Check the `logs` directory for detailed error messages
   - Ensure the database connection is properly configured
   - Verify file permissions in the project directory

3. **Support**:
   - Technical issues: Check the logs directory
   - For access issues: Contact the system administrator
   - For clinical questions: Consult with healthcare professionals

---

**Note**: This system is designed for educational and research purposes and should not replace professional medical advice.
