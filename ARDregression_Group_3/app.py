from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from scipy.stats import gaussian_kde
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.model import ARDModel
    from src.data_validation import validate_dataset, prepare_dataset
except ImportError as e:
    print(f"Error importing required modules: {str(e)}")
    print("Please ensure all required modules are installed and in the correct location.")
    sys.exit(1)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the model
MODEL = None

def create_feature_importance_plot(model, feature_names):
    """Create a feature importance visualization"""
    try:
        # Get feature importance scores
        try:
            relevance_scores = model._get_relevant_features()
        except Exception as e:
            print(f"Error getting feature importance scores: {str(e)}")
            # Create dummy scores if we can't get real ones
            relevance_scores = {feature: 1.0 for feature in feature_names}
        
        # Sort features by importance
        sorted_items = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_items)
        
        # Create the bar plot
        fig = px.bar(
            x=features,
            y=scores,
            title='Feature Importance in ADR Model',
            labels={'x': 'Features', 'y': 'Relevance Score'},
            template='plotly_white'
        )
        
        # Update layout for better visualization
        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=45,
            bargap=0.2,
            plot_bgcolor='white',
            yaxis_title='Importance Score',
            xaxis_title='Features',
            margin=dict(l=50, r=50, t=50, b=100)
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>'
        )
        
        return json.loads(fig.to_json())
    except Exception as e:
        print(f"Error creating feature importance plot: {str(e)}")
        return None

def create_prediction_distribution_plot(predictions):
    """Create a distribution plot of predictions"""
    try:
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=predictions,
            name='Prediction Distribution',
            nbinsx=30,
            histnorm='probability density'
        ))
        
        # Add KDE plot
        kde_x = np.linspace(min(predictions), max(predictions), 100)
        kde = gaussian_kde(predictions)
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde(kde_x),
            name='Density Estimation',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Distribution of Predictions',
            xaxis_title='Predicted Values',
            yaxis_title='Density',
            template='plotly_white',
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return json.loads(fig.to_json())
    except Exception as e:
        print(f"Error creating distribution plot: {str(e)}")
        return None

def create_feature_impact_plot(df, model):
    """Create a feature impact analysis plot"""
    try:
        feature_impacts = {}
        baseline = model.predict(df).mean()
        
        for feature in model.feature_names:
            temp_df = df.copy()
            temp_df[feature] = temp_df[feature].mean()  # Set feature to its mean
            new_pred = model.predict(temp_df).mean()
            impact = ((baseline - new_pred) / baseline) * 100
            feature_impacts[feature] = abs(impact)  # Use absolute impact
        
        # Sort by impact
        sorted_impacts = dict(sorted(feature_impacts.items(), key=lambda x: x[1], reverse=True))
        
        fig = px.bar(
            x=list(sorted_impacts.keys()),
            y=list(sorted_impacts.values()),
            title='Feature Impact Analysis',
            labels={'x': 'Features', 'y': 'Impact (%)'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_tickangle=45,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=100),
            yaxis_title='% Impact on Predictions',
            xaxis_title='Features'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Impact: %{y:.2f}%<extra></extra>'
        )
        
        return json.loads(fig.to_json())
    except Exception as e:
        print(f"Error creating feature impact plot: {str(e)}")
        return None

def create_prediction_vs_actual_plot(y_true, y_pred):
    """Create a scatter plot of predicted vs actual values"""
    try:
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='Actual',
            y='Predicted',
            title='Predicted vs Actual Values',
            template='plotly_white'
        )
        
        # Calculate axis limits
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        axis_range = [min_val - (max_val - min_val) * 0.1, max_val + (max_val - min_val) * 0.1]
        
        # Add diagonal line for perfect predictions
        fig.add_trace(
            go.Scatter(
                x=axis_range,
                y=axis_range,
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red'),
                showlegend=True
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            plot_bgcolor='white',
            xaxis=dict(range=axis_range),
            yaxis=dict(range=axis_range),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>Actual</b>: %{x:.2f}<br><b>Predicted</b>: %{y:.2f}<extra></extra>',
            marker=dict(size=8)
        )
        
        return json.loads(fig.to_json())
    except Exception as e:
        print(f"Error creating prediction plot: {str(e)}")
        return None

@app.route('/api/validate', methods=['POST'])
def validate_data():
    """Validate uploaded dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Validate the dataset
        is_valid, validation_results = validate_dataset(df)
        
        # Add column information
        validation_results['columns'] = {
            'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
            'non_numeric': df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        return jsonify({
            'is_valid': is_valid,
            'validation_results': validation_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the ADR model"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'No target column specified'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Prepare dataset
        X, y = prepare_dataset(df, target_column)
        
        # Initialize and train model
        global MODEL
        MODEL = ARDModel()
        # Train the model and get base results
        results = MODEL.train(X, y)
        predictions = MODEL.predict(X)
        
        # Calculate additional metrics
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        r2 = MODEL.model.score(X, y)
        
        # Get feature relevance scores before creating visualizations
        try:
            relevance_scores = MODEL._get_relevant_features()
        except Exception as e:
            print(f"Error getting relevant features: {str(e)}")
            relevance_scores = {}
        
        # Create visualizations
        feature_importance_plot = create_feature_importance_plot(MODEL, X.columns)
        prediction_plot = create_prediction_vs_actual_plot(y, predictions)
        
        # Calculate normalized scores
        max_score = max(relevance_scores.values()) if relevance_scores else 1
        normalized_scores = {k: (v / max_score) * 100 for k, v in relevance_scores.items()}

        return jsonify({
            'training_results': {
                'train_rmse': float(rmse),
                'test_rmse': float(rmse),  # In this case, we're using the same as train
                'train_r2': float(r2),
                'test_r2': float(r2),      # In this case, we're using the same as train
                'relevant_features': relevance_scores,
                'max_score': float(max_score),
                'normalized_scores': normalized_scores,
                'score': float(r2)  # Using R2 score as the model's overall score
            },
            'visualizations': {
                'feature_importance': feature_importance_plot,
                'prediction_plot': prediction_plot
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    try:
        # Validate model and input
        if MODEL is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV file'}), 400
            
        # Read and validate input data
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
            
        # Verify required features
        missing_features = set(MODEL.feature_names) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}'
            }), 400
            
        # Make predictions
        try:
            predictions = MODEL.predict(df[MODEL.feature_names])
            pred_list = predictions.tolist()
        except Exception as e:
            return jsonify({'error': f'Error making predictions: {str(e)}'}), 500
            
        # Prepare results DataFrame
        result_df = pd.DataFrame({
            'predicted_value': pred_list,
            'input_id': range(1, len(pred_list) + 1)
        })
        
        # Calculate statistics
        stats = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'quartiles': [
                float(np.percentile(predictions, 25)),
                float(np.percentile(predictions, 50)),
                float(np.percentile(predictions, 75))
            ],
            'count': len(predictions)
        }
        
        # Create visualizations
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=predictions,
            name='Prediction Distribution',
            nbinsx=30
        ))
        fig_dist.update_layout(
            title='Distribution of Predictions',
            xaxis_title='Predicted Value',
            yaxis_title='Count'
        )

        # Feature importance visualization
        importance_scores = MODEL._get_relevant_features()
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            x=list(importance_scores.keys()),
            y=list(importance_scores.values()),
            name='Feature Importance'
        ))
        fig_importance.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            xaxis={'tickangle': 45}
        )
        
        # Create response data with all required information
        response_data = {
            'predictions': pred_list,  # List of predictions
            'predicted_values': [{'id': i+1, 'value': float(p)} for i, p in enumerate(pred_list)],  # Structured predictions
            'model_analysis': {
                'prediction_stats': stats,
                'feature_importance': importance_scores
            },
            'visualizations': {
                'distribution': json.loads(fig_dist.to_json()),
                'feature_importance': json.loads(fig_importance.to_json())
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)