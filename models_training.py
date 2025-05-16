"""
Enhanced Diabetes Prediction Model Training Script
- Data Preprocessing
- Feature Selection
- Model Training with Hyperparameter Tuning
- Model Evaluation
- Model Interpretability (SHAP, LIME & Permutation Importance)
- Model Export
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                            accuracy_score, f1_score, precision_score, recall_score, 
                            precision_recall_curve, average_precision_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import shap
import lime
from lime import lime_tabular
from log import (
    log_training_start, 
    log_training_epoch, 
    log_training_end, 
    log_cross_validation
)

# Suppress warnings (keeping for other potential warnings)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up output directories
os.makedirs('model_outputs', exist_ok=True)
os.makedirs('model_plots', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load diabetes dataset and perform preprocessing
    """
    print("\n[1] LOADING AND PREPROCESSING DATA")
    
    # Load the data
    df = pd.read_csv(file_path)
    print(f"Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Make a copy for preprocessing
    data = df.copy()
    
    # Check for columns with zero values (potential missing values)
    print("\nChecking for zero values (potential missing values):")
    cols_with_zeros = {}
    for col in data.columns:
        if col not in ['Id', 'Outcome']:
            zero_count = (data[col] == 0).sum()
            if zero_count > 0:
                cols_with_zeros[col] = zero_count
                print(f"- {col}: {zero_count} zeros ({zero_count/len(data)*100:.2f}%)")
    
    # Feature engineering based on EDA insights
    print("\nPerforming feature engineering:")
    
    # Handle zeros in medical features (replace with median of non-zero values)
    medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in medical_features:
        if col in cols_with_zeros:
            # Get median of non-zero values
            median_val = data.loc[data[col] > 0, col].median()
            # Replace zeros with median
            data.loc[data[col] == 0, col] = median_val
            print(f"- Replaced zeros in {col} with median: {median_val:.2f}")
    
    # Create derived features
    print("\nCreating derived features:")
    
    # Glucose to Insulin ratio
    data['Glucose_to_Insulin_Ratio'] = data['Glucose'] / (data['Insulin'] + 1)  # Avoid division by zero
    print("- Created Glucose_to_Insulin_Ratio")
    
    # BMI * Age interaction
    data['BMI_Age_Product'] = data['BMI'] * data['Age'] / 100
    print("- Created BMI_Age_Product")
    
    # Binary indicators for high-risk factors
    data['Has_High_Glucose'] = (data['Glucose'] > 140).astype(int)
    data['Has_High_BP'] = (data['BloodPressure'] > 90).astype(int)
    data['Has_High_BMI'] = (data['BMI'] > 30).astype(int)
    print("- Created binary indicators for high-risk factors")
    
    # Log transform of skewed features
    skewed_features = ['Insulin', 'DiabetesPedigreeFunction']
    for feature in skewed_features:
        data[f'Log_{feature}'] = np.log1p(data[feature])
        print(f"- Created log transform of {feature}")
    
    # Split data into features and target
    X = data.drop(['Outcome', 'Id'] if 'Id' in data.columns else ['Outcome'], axis=1)
    y = data['Outcome']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"\nData split into train and test sets:")
    print(f"- Training set: {X_train.shape[0]} samples")
    print(f"- Test set: {X_test.shape[0]} samples")
    
    # Save the feature names for later use with SHAP and LIME
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names

# Function to plot learning curves
def plot_learning_curves(estimator, X, y, cv, scoring='accuracy', model_name='model'):
    """
    Generate and plot learning curves for a model
    """
    print(f"\nGenerating learning curves for {model_name}...")
    
    # Calculate learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=train_sizes, n_jobs=-1, random_state=RANDOM_SEED
    )
    
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(12, 8))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
    
    plt.xlabel('Training set size', fontsize=14)
    plt.ylabel(f'{scoring.capitalize()} Score', fontsize=14)
    plt.title(f'Learning Curves - {model_name}', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'model_plots/{model_name}_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot feature importance
def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance from model
    """
    # Extract feature importance
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        # Get feature importance from pipeline
        classifier = model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            importance = classifier.feature_importances_
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute.")
            return
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print(f"Model {model_name} does not have feature_importances_ attribute.")
        return
    
    # Create DataFrame for plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Save to CSV
    feature_importance.to_csv(f'model_outputs/{model_name}_feature_importance.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette='viridis')
    plt.title(f'Top 15 Feature Importance - {model_name}', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'model_plots/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot precision-recall curve
def plot_precision_recall_curve(model, X_test, y_test, model_name):
    """
    Generate and plot precision-recall curve
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision and recall for different threshold values
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate average precision score
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f'AP={avg_precision:.3f}')
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'model_plots/{model_name}_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot ROC curves for multiple models
def plot_combined_roc_curves(models_dict, X_test, y_test):
    """
    Generate and plot ROC curves for multiple models in one figure
    """
    plt.figure(figsize=(12, 10))
    
    for model_name, model in models_dict.items():
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot random prediction line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves Comparison', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_plots/combined_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot threshold analysis
def plot_threshold_analysis(model, X_test, y_test, model_name):
    """
    Plot precision, recall, and F1 score for different threshold values
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precision_scores, 'b-', label='Precision')
    plt.plot(thresholds, recall_scores, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(f'Threshold Analysis - {model_name}', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'model_plots/{model_name}_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to analyze training history
def analyze_training_history(model, X_train, y_train, model_name):
    """
    Train model with evaluation set to get training history - Fixed for XGBoost compatibility
    """
    if model_name != 'xgboost':
        print(f"Training history analysis only available for XGBoost model.")
        return
    try:
        # Get classifier from pipeline
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
        else:
            classifier = model
            
        # Extract best parameters
        params = classifier.get_params()
        
        # Remove parameters that aren't actual XGBoost parameters or cause conflicts
        params_to_remove = ['base_score', 'callbacks', 'early_stopping_rounds', 'enable_categorical', 
                           'eval_metric', 'feature_types', 'importance_type', 'kwargs', 
                           'missing', 'n_estimators_', 'n_jobs', 'use_label_encoder', 'random_state']
        for param in params_to_remove:
            params.pop(param, None)
                
        # Add evaluation metrics to parameters
        params['eval_metric'] = ['error', 'logloss', 'auc']
        
        # Create evaluation set
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Initialize new model with updated parameters
        new_model = XGBClassifier(**params, random_state=RANDOM_SEED)
        
        # Train model without early_stopping_rounds
        new_model.fit(
            X_train_sub, y_train_sub,
            eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
            verbose=False
        )
        
        # Plot training history
        results = new_model.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(14, 10))
        
        # Classification Error
        plt.subplot(2, 2, 1)
        plt.plot(x_axis, results['validation_0']['error'], label='Train')
        plt.plot(x_axis, results['validation_1']['error'], label='Validation')
        plt.title('XGBoost Training - Classification Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        
        # Log Loss
        plt.subplot(2, 2, 2)
        plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
        plt.title('XGBoost Training - Log Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # AUC
        plt.subplot(2, 2, 3)
        plt.plot(x_axis, results['validation_0']['auc'], label='Train')
        plt.plot(x_axis, results['validation_1']['auc'], label='Validation')
        plt.title('XGBoost Training - AUC')
        plt.xlabel('Iterations')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(2, 2, 4)
        train_error = np.array(results['validation_0']['error'])
        val_error = np.array(results['validation_1']['error'])
        plt.plot(x_axis, 1 - train_error, label='Train Accuracy')
        plt.plot(x_axis, 1 - val_error, label='Validation Accuracy')
        plt.title('XGBoost Training - Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'model_plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"XGBoost training history analysis completed and saved.")
        
    except Exception as e:
        print(f"Error in XGBoost training history analysis: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        print("Proceeding without training history analysis")

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, feature_names, model_name):
    """
    Evaluate model performance with various metrics and visualizations
    """
    print(f"\n[3] EVALUATING {model_name.upper()} PERFORMANCE")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nPerformance Metrics:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- ROC AUC: {roc_auc:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    })
    metrics_df.to_csv(f'model_outputs/{model_name}_metrics.csv', index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'model_plots/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {model_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'model_plots/{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot precision-recall curve
    plot_precision_recall_curve(model, X_test, y_test, model_name)
    
    # Plot threshold analysis
    plot_threshold_analysis(model, X_test, y_test, model_name)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names, model_name)
    
    return accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba

# Function for SHAP analysis
def apply_shap(model, X_test, feature_names, model_name):
    """
    Apply SHAP for any tree-based model with proper array handling
    """
    print(f"\n[4] APPLYING SHAP FOR {model_name.upper()}")
    
    try:
        import traceback  # Add import at the top
        
        # Sample data for SHAP (convert to numpy array if needed)
        X_sample = X_test.sample(min(100, len(X_test)), random_state=RANDOM_SEED)
        
        # Extract classifier and transform data
        if hasattr(model, 'named_steps'):
            if 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                # Transform data through pipeline
                X_transformed = model.named_steps['scaler'].transform(X_sample)
                # Convert to DataFrame to maintain feature names
                X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        else:
            classifier = model
            X_transformed = X_sample

        # Ensure data is 2D
        if len(X_transformed.shape) == 1:
            X_transformed = X_transformed.reshape(1, -1)
            
        # Create explainer with raw output for all models
        explainer = shap.TreeExplainer(classifier, model_output='raw')
            
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_transformed)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification with two-class output
            shap_values = shap_values[1]  # Take positive class
        elif len(shap_values.shape) == 3:
            # For binary classification with single multi-dimensional output
            shap_values = shap_values[:, :, 1]  # Take positive class
            
        # Ensure shap_values are 2D
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
            
        # Calculate mean absolute SHAP values for feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Create and save feature importance DataFrame
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': mean_shap_values
        }).sort_values('SHAP_Importance', ascending=False)
        
        # Plot SHAP-based feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='SHAP_Importance', y='Feature', data=shap_importance.head(15))
        plt.title(f'SHAP-based Feature Importance - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'model_plots/{model_name}_shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save SHAP importance to CSV
        shap_importance.to_csv(f'model_outputs/{model_name}_shap_importance.csv', index=False)
        print(f"SHAP analysis completed for {model_name}")
        
        # Create summary plot for XGBoost
        if model_name.lower() == 'xgboost':
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_transformed,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig(f'model_plots/{model_name}_shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Full error details: {e.__dict__}")
        traceback.print_exc()  # Print full traceback
        print("Proceeding without SHAP analysis")

# Function to apply permutation importance (alternative to SHAP)
def apply_permutation_importance(model, X_test, y_test, feature_names, model_name):
    """
    Calculate and plot permutation feature importance
    """
    print(f"\n[4] APPLYING PERMUTATION IMPORTANCE FOR {model_name.upper()}")
    
    try:
        # Ensure X_test is a DataFrame for permutation importance
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
        )
        
        # Create DataFrame for analysis
        perm_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values('Importance', ascending=False)
        
        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=perm_importance.head(15))
        plt.title(f'Permutation Feature Importance - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'model_plots/{model_name}_permutation_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save to file
        perm_importance.to_csv(f'model_outputs/{model_name}_permutation_importance.csv', index=False)
        
        print(f"Permutation importance analysis completed for {model_name}")
        
    except Exception as e:
        print(f"Error in permutation importance: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Proceeding without permutation importance analysis")

# Function to apply LIME for model interpretability
def apply_lime(model, X_train, X_test, feature_names, model_name):
    """
    Apply LIME to explain individual predictions
    """
    print(f"\n[5] APPLYING LIME FOR {model_name.upper()} INTERPRETABILITY")
    
    try:
        # Ensure X_train is a DataFrame with feature names
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        # Create a LIME explainer with scaled data
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            X_train_scaled = scaler.transform(X_train)
        else:
            scaler = StandardScaler() if model_name == 'xgboost' else RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        
        lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train_scaled,
            feature_names=feature_names,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification',
            random_state=RANDOM_SEED
        )
        
        # Custom predict function to handle scaling
        def predict_fn(X):
            if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                return model.predict_proba(X)
            else:
                X_scaled = scaler.transform(X)
                return model.predict_proba(X_scaled)
        
        # Sample a few test instances to explain
        num_explanations = min(5, len(X_test))
        
        for i in range(num_explanations):
            # Scale the instance
            instance = X_test.iloc[i].values.reshape(1, -1)
            if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                instance_scaled = instance
            else:
                instance_scaled = scaler.transform(instance)
            
            # Get explanation for the scaled instance
            exp = lime_explainer.explain_instance(
                instance_scaled[0], predict_fn, num_features=10
            )
            
            # Plot explanation
            plt.figure(figsize=(12, 8))
            exp.as_pyplot_figure()
            plt.title(f'LIME Explanation for Instance {i+1} - {model_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'model_plots/{model_name}_lime_explanation_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"LIME analysis completed and saved for {model_name}")
        
    except Exception as e:
        print(f"Error in LIME analysis: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        print("Proceeding without LIME analysis")

# Function to train and optimize models
def train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train, optimize and evaluate multiple models
    """
    print("\n[2] TRAINING AND OPTIMIZING MODELS")
    
    # Create a dictionary to store results
    model_results = {}
    models_dict = {}
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Ensure X_train and X_test are DataFrames with feature names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    # 1. Random Forest
    print("\nTraining Random Forest model with hyperparameter optimization...")
    
    # Define pipeline with preprocessing
    rf_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', RandomForestClassifier(random_state=RANDOM_SEED))
    ])
    
    # Define hyperparameter grid
    rf_param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Log training start for Random Forest
    log_training_start('random_forest', rf_param_grid)
    
    # Set up grid search for Random Forest
    rf_grid = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1,
        return_train_score=True
    )
    
    # Train model with grid search
    start_time = time.time()
    rf_grid.fit(X_train, y_train)
    rf_training_time = time.time() - start_time
    
    # Get best model
    rf_best_model = rf_grid.best_estimator_
    
    # Log cross-validation results for Random Forest
    log_cross_validation('random_forest', 0, {'mean_f1': rf_grid.best_score_})
    
    print(f"Random Forest training completed in {rf_training_time:.2f} seconds")
    print(f"Best parameters: {rf_grid.best_params_}")
    print(f"Best cross-validation score: {rf_grid.best_score_:.4f}")
    
    # Plot learning curves
    plot_learning_curves(
        rf_best_model, X_train, y_train, cv=cv, 
        scoring='accuracy', model_name='random_forest'
    )
    
    # Evaluate model
    rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc, rf_y_pred, rf_y_pred_proba = evaluate_model(
        rf_best_model, X_test, y_test, feature_names, "random_forest"
    )
    
    # Log training completion for Random Forest
    rf_metrics = {
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1': rf_f1,
        'auc': rf_roc_auc
    }
    log_training_end('random_forest', rf_metrics, rf_training_time)
    
    # Apply SHAP
    apply_shap(rf_best_model, X_test, feature_names, "random_forest")
    
    # Apply permutation importance
    apply_permutation_importance(rf_best_model, X_test, y_test, feature_names, "random_forest")
    
    # Apply LIME
    apply_lime(rf_best_model, X_train, X_test, feature_names, "random_forest")
    
    # Store results
    model_results['random_forest'] = {
        'model': rf_best_model,
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1': rf_f1,
        'auc': rf_roc_auc,
        'training_time': rf_training_time,
        'best_params': rf_grid.best_params_
    }
    
    models_dict['Random Forest'] = rf_best_model
    
    # 2. Decision Tree
    print("\nTraining Decision Tree model with hyperparameter optimization...")
    
    # Define pipeline with preprocessing
    dt_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', DecisionTreeClassifier(random_state=RANDOM_SEED))
    ])
    
    # Define hyperparameter grid
    dt_param_grid = {
        'classifier__max_depth': [None, 5, 10, 15, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__criterion': ['gini', 'entropy']
    }
    
    # Log training start for Decision Tree
    log_training_start('decision_tree', dt_param_grid)
    
    # Set up grid search for Decision Tree
    dt_grid = GridSearchCV(
        dt_pipeline, dt_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1,
        return_train_score=True
    )
    
    # Train model with grid search
    start_time = time.time()
    dt_grid.fit(X_train, y_train)
    dt_training_time = time.time() - start_time
    
    # Get best model
    dt_best_model = dt_grid.best_estimator_
    
    # Log cross-validation results for Decision Tree
    log_cross_validation('decision_tree', 0, {'mean_f1': dt_grid.best_score_})
    
    print(f"Decision Tree training completed in {dt_training_time:.2f} seconds")
    print(f"Best parameters: {dt_grid.best_params_}")
    print(f"Best cross-validation score: {dt_grid.best_score_:.4f}")
    
    # Plot learning curves
    plot_learning_curves(
        dt_best_model, X_train, y_train, cv=cv, 
        scoring='accuracy', model_name='decision_tree'
    )
    
    # Evaluate model
    dt_accuracy, dt_precision, dt_recall, dt_f1, dt_roc_auc, dt_y_pred, dt_y_pred_proba = evaluate_model(
        dt_best_model, X_test, y_test, feature_names, "decision_tree"
    )
    
    # Log training completion for Decision Tree
    dt_metrics = {
        'accuracy': dt_accuracy,
        'precision': dt_precision,
        'recall': dt_recall,
        'f1': dt_f1,
        'auc': dt_roc_auc
    }
    log_training_end('decision_tree', dt_metrics, dt_training_time)
    
    # Apply SHAP
    apply_shap(dt_best_model, X_test, feature_names, "decision_tree")
    
    # Apply permutation importance
    apply_permutation_importance(dt_best_model, X_test, y_test, feature_names, "decision_tree")
    
    # Apply LIME
    apply_lime(dt_best_model, X_train, X_test, feature_names, "decision_tree")
    
    # Store results
    model_results['decision_tree'] = {
        'model': dt_best_model,
        'accuracy': dt_accuracy,
        'precision': dt_precision,
        'recall': dt_recall,
        'f1': dt_f1,
        'auc': dt_roc_auc,
        'training_time': dt_training_time,
        'best_params': dt_grid.best_params_
    }
    
    models_dict['Decision Tree'] = dt_best_model
    
    # 3. XGBoost
    print("\nTraining XGBoost model with hyperparameter optimization...")
    
    # Define pipeline with preprocessing
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=RANDOM_SEED))
    ])
    
    # Define hyperparameter grid
    xgb_param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Log training start for XGBoost
    log_training_start('xgboost', xgb_param_grid)
    
    # Set up grid search for XGBoost
    xgb_grid = GridSearchCV(
        xgb_pipeline, xgb_param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1,
        return_train_score=True
    )
    
    # Train model with grid search
    start_time = time.time()
    xgb_grid.fit(X_train, y_train)
    xgb_training_time = time.time() - start_time
    
    # Get best model
    xgb_best_model = xgb_grid.best_estimator_
    
    # Log cross-validation results for XGBoost
    log_cross_validation('xgboost', 0, {'mean_f1': xgb_grid.best_score_})
    
    print(f"XGBoost training completed in {xgb_training_time:.2f} seconds")
    print(f"Best parameters: {xgb_grid.best_params_}")
    print(f"Best cross-validation score: {xgb_grid.best_score_:.4f}")
    
    # Plot learning curves
    plot_learning_curves(
        xgb_best_model, X_train, y_train, cv=cv, 
        scoring='accuracy', model_name='xgboost'
    )
    
    # Plot XGBoost training history
    analyze_training_history(xgb_best_model, X_train, y_train, 'xgboost')
    
    # Evaluate model
    xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_roc_auc, xgb_y_pred, xgb_y_pred_proba = evaluate_model(
        xgb_best_model, X_test, y_test, feature_names, "xgboost"
    )
    
    # Log training completion for XGBoost
    xgb_metrics = {
        'accuracy': xgb_accuracy,
        'precision': xgb_precision,
        'recall': xgb_recall,
        'f1': xgb_f1,
        'auc': xgb_roc_auc
    }
    log_training_end('xgboost', xgb_metrics, xgb_training_time)
    
    # Apply SHAP
    apply_shap(xgb_best_model, X_test, feature_names, "xgboost")
    
    # Apply permutation importance
    apply_permutation_importance(xgb_best_model, X_test, y_test, feature_names, "xgboost")
    
    # Apply LIME
    apply_lime(xgb_best_model, X_train, X_test, feature_names, "xgboost")
    
    # Store results
    model_results['xgboost'] = {
        'model': xgb_best_model,
        'accuracy': xgb_accuracy,
        'precision': xgb_precision,
        'recall': xgb_recall,
        'f1': xgb_f1,
        'auc': xgb_roc_auc,
        'training_time': xgb_training_time,
        'best_params': xgb_grid.best_params_
    }
    
    models_dict['XGBoost'] = xgb_best_model
    
    # Plot combined ROC curves
    plot_combined_roc_curves(models_dict, X_test, y_test)
    
    # Compare model performance
    print("\n[6] COMPARING MODEL PERFORMANCE")
    
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'Decision Tree', 'XGBoost'],
        'Accuracy': [rf_accuracy, dt_accuracy, xgb_accuracy],
        'Precision': [rf_precision, dt_precision, xgb_precision],
        'Recall': [rf_recall, dt_recall, xgb_recall],
        'F1 Score': [rf_f1, dt_f1, xgb_f1],
        'ROC AUC': [rf_roc_auc, dt_roc_auc, xgb_roc_auc],
        'Training Time (s)': [rf_training_time, dt_training_time, xgb_training_time]
    })
    
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save comparison to file
    comparison_df.to_csv('model_outputs/model_comparison.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    x = np.arange(len(metrics))
    width = 0.25  # Reduced width to accommodate three bars
    
    plt.bar(x - width, comparison_df[metrics].iloc[0], width, label='Random Forest')
    plt.bar(x, comparison_df[metrics].iloc[1], width, label='Decision Tree')
    plt.bar(x + width, comparison_df[metrics].iloc[2], width, label='XGBoost')
    
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a radar chart for model comparison
    plt.figure(figsize=(10, 10))
    
    # Data for radar chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    n = len(categories)
    
    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add values for Random Forest
    rf_values = comparison_df.iloc[0][categories].tolist()
    rf_values += rf_values[:1]  # Close the loop
    
    # Add values for Decision Tree
    dt_values = comparison_df.iloc[1][categories].tolist()
    dt_values += dt_values[:1]  # Close the loop
    
    # Add values for XGBoost
    xgb_values = comparison_df.iloc[2][categories].tolist()
    xgb_values += xgb_values[:1]  # Close the loop
    
    # Add category labels to chart
    ax = plt.subplot(111, polar=True)
    
    # Add category labels
    plt.xticks(angles[:-1], categories)
    
    # Plot data
    ax.plot(angles, rf_values, 'b-', linewidth=2, label='Random Forest')
    ax.fill(angles, rf_values, 'b', alpha=0.1)
    
    ax.plot(angles, dt_values, 'g-', linewidth=2, label='Decision Tree')
    ax.fill(angles, dt_values, 'g', alpha=0.1)
    
    ax.plot(angles, xgb_values, 'r-', linewidth=2, label='XGBoost')
    ax.fill(angles, xgb_values, 'r', alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.title('Model Performance Radar Chart', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_plots/model_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Determine the best model based on F1 score
    best_f1 = rf_f1
    best_model_name = 'random_forest'
    
    if dt_f1 > best_f1:
        best_f1 = dt_f1
        best_model_name = 'decision_tree'
    
    if xgb_f1 > best_f1:
        best_f1 = xgb_f1
        best_model_name = 'xgboost'
    
    print(f"\nBest performing model based on F1 score: {best_model_name}")
    
    return model_results, best_model_name

# Function to save models
def save_models(model_results):
    """
    Save trained models to disk
    """
    print("\n[7] SAVING MODELS")
    
    for model_name, results in model_results.items():
        model = results['model']
        
        # Save model
        with open(f'saved_models/{model_name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results.get('auc', 0),
            'training_time': results['training_time'],
            'best_params': results['best_params']
        }
        
        with open(f'saved_models/{model_name}_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model {model_name} saved successfully")
    
    print("All models saved successfully")

# Main function
def main():
    """
    Main function to run the entire pipeline
    """
    print("="*80)
    print("                DIABETES PREDICTION MODEL TRAINING")
    print("="*80)
    
    # Set path to data
    file_path = os.path.join('data', 'Healthcare-Diabetes.csv')
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)
    
    # Train and evaluate models
    model_results, best_model_name = train_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Save models
    save_models(model_results)
    
    print("\n[8] TRAINING PIPELINE COMPLETED")
    print(f"Best model: {best_model_name}")
    print(f"Model files saved in 'saved_models' directory")
    print(f"Evaluation metrics saved in 'model_outputs' directory")
    print(f"Plots saved in 'model_plots' directory")

# Run the main function
if __name__ == "__main__":
    main()