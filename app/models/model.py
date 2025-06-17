import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class ModelResults:
    """Results from model training and evaluation"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None

class NFTAuthenticityModel:
    """
    ML Model for detecting NFT authenticity and scam probability
    """
    
    def __init__(self, data_path: str = "training_data/nft_features_latest.json"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        self.results = {}
        
        # Define feature groups for better analysis
        self.feature_groups = {
            'collection_metrics': [
                'total_supply', 'floor_price', 'market_cap', 
                'total_volume', 'num_owners', 'average_price'
            ],
            'verification_features': [
                'is_verified', 'has_discord', 'has_twitter',
                'trait_offers_enabled', 'collection_offers_enabled'
            ],
            'social_features': [
                'reddit_mentions', 'reddit_sentiment', 'reddit_engagement'
            ]
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data and prepare it for ML
        """
        print("Loading training data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training data not found: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Extract features
        features_list = []
        for record in data.get('features', []):
            features_list.append(record)
        
        if not features_list:
            raise ValueError("No features found in training data")
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Store feature names (exclude non-feature columns)
        exclude_cols = ['collection_slug', 'token_id']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Loaded {len(df)} records with {len(self.feature_names)} features")
        print(f"Features: {self.feature_names}")
        
        # For now, create synthetic labels based on heuristics
        # In a real scenario, you'd have manually labeled data
        y = self.create_synthetic_labels(df)
        X = df[self.feature_names]
        
        return X, y
    
    def create_synthetic_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create synthetic labels for demonstration
        In production, you'd have manually labeled data
        """
        print("Creating synthetic labels based on heuristics...")
        
        labels = []
        for _, row in df.iterrows():
            score = 0
            
            # Verified collections are more trustworthy
            if row.get('is_verified', 0) == 1:
                score += 3
            
            # Collections with social presence are more trustworthy
            if row.get('has_discord', 0) == 1:
                score += 1
            if row.get('has_twitter', 0) == 1:
                score += 1
            
            # Higher volume and owners suggest legitimacy
            if row.get('total_volume', 0) > 10000:  # > 10K ETH volume
                score += 2
            if row.get('num_owners', 0) > 1000:  # > 1000 owners
                score += 2
            
            # Floor price stability (not too high, not too low)
            floor_price = row.get('floor_price', 0)
            if 0.1 <= floor_price <= 100:  # Reasonable floor price range
                score += 1
            
            # Social engagement
            if row.get('reddit_mentions', 0) > 0:
                score += 1
            
            # Label as legitimate if score >= 5, otherwise suspicious
            labels.append(1 if score >= 5 else 0)  # 1 = legitimate, 0 = suspicious
        
        labels = pd.Series(labels, name='is_legitimate')
        
        print(f"Label distribution:")
        print(f"  Legitimate (1): {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"  Suspicious (0): {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
        
        return labels
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering and preprocessing
        """
        print("Preparing features...")
        
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = X_processed.fillna(0)
        
        # Create derived features
        
        # 1. Market efficiency ratio
        X_processed['market_efficiency'] = np.where(
            X_processed['total_supply'] > 0,
            X_processed['num_owners'] / X_processed['total_supply'],
            0
        )
        
        # 2. Volume per owner
        X_processed['volume_per_owner'] = np.where(
            X_processed['num_owners'] > 0,
            X_processed['total_volume'] / X_processed['num_owners'],
            0
        )
        
        # 3. Price to volume ratio
        X_processed['price_volume_ratio'] = np.where(
            X_processed['total_volume'] > 0,
            X_processed['floor_price'] / X_processed['total_volume'],
            0
        )
        
        # 4. Social engagement score
        X_processed['social_score'] = (
            X_processed['has_discord'] + 
            X_processed['has_twitter'] + 
            np.where(X_processed['reddit_mentions'] > 0, 1, 0)
        )
        
        # 5. Verification score
        X_processed['verification_score'] = (
            X_processed['is_verified'] * 3 +
            X_processed['trait_offers_enabled'] +
            X_processed['collection_offers_enabled']
        )
        
        # Handle infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], 0)
        
        # Update feature names
        self.feature_names = list(X_processed.columns)
        
        print(f"Created {len(self.feature_names)} features after engineering")
        
        return X_processed
    
    def train_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Train multiple ML models
        """
        print("\nTraining multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for algorithms that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=42, 
                probability=True, 
                class_weight='balanced'
            )
        }
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for certain models
            if name in ['Logistic Regression', 'SVM']:
                X_train_used = X_train_scaled
                X_test_used = X_test_scaled
            else:
                X_train_used = X_train
                X_test_used = X_test
            
            # Train model
            model.fit(X_train_used, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_used)
            y_pred_proba = model.predict_proba(X_test_used)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC AUC (handle case where only one class is present)
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                roc_auc = 0.5  # Default for single class
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.feature_names, 
                    model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(
                    self.feature_names, 
                    abs(model.coef_[0])
                ))
            
            # Store results
            self.results[name] = ModelResults(
                model_name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                confusion_matrix=cm,
                feature_importance=feature_importance
            )
            
            # Store model
            self.models[name] = model
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            print(f"  ROC AUC: {roc_auc:.3f}")
    
    def evaluate_models(self):
        """
        Print detailed evaluation of all models
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1_score,
                'ROC AUC': result.roc_auc
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Find best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best performing model: {best_model_name}")
        
        return best_model_name
    
    def plot_results(self, save_dir: str = "model_results"):
        """
        Create visualizations of model performance
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Model Comparison Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            values = [getattr(self.results[model], metric) for model in models]
            
            bars = ax.bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax.set_title(f'{name} by Model')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance Plot (for Random Forest)
        if 'Random Forest' in self.results and self.results['Random Forest'].feature_importance:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            importance = self.results['Random Forest'].feature_importance
            features = list(importance.keys())
            values = list(importance.values())
            
            # Sort by importance
            sorted_data = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_data)
            
            # Plot top 15 features
            top_n = min(15, len(features))
            y_pos = np.arange(top_n)
            
            ax.barh(y_pos, values[:top_n], color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features[:top_n])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top Feature Importances (Random Forest)')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_models(self, save_dir: str = "saved_models"):
        """
        Save trained models and metadata
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = f"{save_dir}/{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = f"{save_dir}/{name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            print(f"Saved {name} scaler to {scaler_path}")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_groups': self.feature_groups,
            'model_results': {
                name: {
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'roc_auc': result.roc_auc
                }
                for name, result in self.results.items()
            },
            'training_date': datetime.now().isoformat(),
            'data_path': self.data_path
        }
        
        metadata_path = f"{save_dir}/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    def predict_collection(self, collection_features: Dict, model_name: str = 'Random Forest') -> Dict:
        """
        Predict authenticity for a new collection
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare features
        df = pd.DataFrame([collection_features])
        df = df.reindex(columns=self.feature_names, fill_value=0)
        df = self.prepare_features(df)
        
        # Scale if needed
        if model_name in ['Logistic Regression', 'SVM']:
            scaler = self.scalers['standard']
            features = scaler.transform(df)
        else:
            features = df
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return {
            'is_legitimate': bool(prediction),
            'legitimacy_probability': float(probability[1]),
            'risk_score': float(1 - probability[1]),
            'confidence': float(max(probability)),
            'model_used': model_name
        }

def main():
    """
    Main function to train the NFT authenticity model
    """
    print("🚀 NFT Authenticity Model Training")
    print("="*50)
    
    # Initialize model
    model = NFTAuthenticityModel()
    
    try:
        # Load and prepare data
        X, y = model.load_and_prepare_data()
        X_processed = model.prepare_features(X)
        
        # Train models
        model.train_models(X_processed, y)
        
        # Evaluate models
        best_model = model.evaluate_models()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        model.plot_results()
        
        # Save models
        print("\n💾 Saving models...")
        model.save_models()
        
        print(f"\n✅ Training complete! Best model: {best_model}")
        print("\n📁 Check the following directories:")
        print("  - model_results/ for visualizations")
        print("  - saved_models/ for trained models")
        
        # Example prediction
        print("\n🔮 Example prediction:")
        example_features = {
            'total_supply': 10000,
            'is_verified': 1,
            'has_discord': 1,
            'has_twitter': 1,
            'trait_offers_enabled': 1,
            'collection_offers_enabled': 1,
            'floor_price': 5.0,
            'market_cap': 50000,
            'total_volume': 100000,
            'num_owners': 3000,
            'average_price': 10.0,
            'reddit_mentions': 5,
            'reddit_sentiment': 0.7,
            'reddit_engagement': 10
        }
        
        prediction = model.predict_collection(example_features, best_model)
        print(f"  Legitimacy: {'✅ Legitimate' if prediction['is_legitimate'] else '⚠️ Suspicious'}")
        print(f"  Confidence: {prediction['confidence']:.1%}")
        print(f"  Risk Score: {prediction['risk_score']:.1%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
