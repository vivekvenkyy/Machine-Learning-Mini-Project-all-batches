from sklearn.linear_model import ARDRegression
import numpy as np

class ARDModel:
    def __init__(self):
        self.model = ARDRegression(n_iter=300)
        self.feature_names = None
        
    def train(self, X, y):
        """Train the ARD model"""
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        
        # Calculate metrics
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        r2 = self.model.score(X, y)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'n_iter': int(self.model.n_iter_),
            'lambda': float(self.model.lambda_),
            'alpha': float(self.model.alpha_)
        }
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if isinstance(X, np.ndarray):
            return self.model.predict(X)
        return self.model.predict(X[self.feature_names])
    
    def _get_relevant_features(self):
        """Get feature importance scores"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model has not been trained yet")
            
        relevance = {feature: abs(coef) for feature, coef in zip(self.feature_names, self.model.coef_)}
        return dict(sorted(relevance.items(), key=lambda x: abs(x[1]), reverse=True))
