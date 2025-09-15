
"""
JPMorgan Chase Credit Risk Model Training & Evaluation
=====================================================

Enterprise-grade machine learning pipeline for credit risk assessment
with comprehensive evaluation metrics and business-focused insights.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, accuracy_score,
                            precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class CreditRiskModelTrainer:
    """Comprehensive credit risk model training and evaluation"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.performance_metrics = {}
        self.feature_importance = {}

    def train_models(self, X, y, test_size=0.2):
        """Train multiple credit risk models"""

        print("🚀 Training Credit Risk Models")
        print("=" * 50)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training default rate: {y_train.mean():.3%}")
        print(f"Test default rate: {y_test.mean():.3%}")

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=15,
                min_samples_split=100,
                min_samples_leaf=50,
                class_weight='balanced'
            )
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\n🔄 Training {model_name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate comprehensive metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='roc_auc'
            )
            metrics['cv_roc_auc_mean'] = cv_scores.mean()
            metrics['cv_roc_auc_std'] = cv_scores.std()

            # Store model and metrics
            self.models[model_name] = {
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'X_test': X_test
            }
            self.performance_metrics[model_name] = metrics

            # Feature importance (for Random Forest)
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[model_name] = feature_importance

            # Print performance summary
            print(f"✅ {model_name} Performance:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']*2:.4f})")

        return X_train, X_test, y_train, y_test

    def plot_model_comparison(self):
        """Create comprehensive model comparison visualizations"""

        print("\n📊 Creating model comparison visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('JPMorgan Chase Credit Risk Model Performance Comparison', 
                     fontsize=16, fontweight='bold')

        # 1. ROC Curves
        ax1 = axes[0, 0]
        for model_name, model_data in self.models.items():
            fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'])
            auc = self.performance_metrics[model_name]['roc_auc']
            ax1.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        for model_name, model_data in self.models.items():
            precision, recall, _ = precision_recall_curve(
                model_data['y_test'], model_data['y_pred_proba']
            )
            ax2.plot(recall, precision, linewidth=2, label=f'{model_name}')

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Performance Metrics Comparison
        ax3 = axes[0, 2]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.models.keys())

        x = np.arange(len(metrics))
        width = 0.35

        for i, model_name in enumerate(model_names):
            values = [self.performance_metrics[model_name][metric] for metric in metrics]
            ax3.bar(x + i*width, values, width, label=model_name, alpha=0.8)

        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Confusion Matrix for Logistic Regression
        if 'Logistic Regression' in self.models:
            ax4 = axes[1, 0]
            cm = self.performance_metrics['Logistic Regression']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
            ax4.set_title('Confusion Matrix - Logistic Regression')
            ax4.set_ylabel('Actual')
            ax4.set_xlabel('Predicted')

        # 5. Confusion Matrix for Random Forest
        if 'Random Forest' in self.models:
            ax5 = axes[1, 1]
            cm = self.performance_metrics['Random Forest']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax5)
            ax5.set_title('Confusion Matrix - Random Forest')
            ax5.set_ylabel('Actual')
            ax5.set_xlabel('Predicted')

        # 6. Feature Importance (Random Forest)
        if 'Random Forest' in self.feature_importance:
            ax6 = axes[1, 2]
            top_features = self.feature_importance['Random Forest'].head(10)
            ax6.barh(top_features['feature'], top_features['importance'])
            ax6.set_title('Top 10 Feature Importance - Random Forest')
            ax6.set_xlabel('Importance')

        plt.tight_layout()
        plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Model comparison visualizations created and saved to reports/model_comparison.png")

    def generate_business_insights(self):
        """Generate business-focused insights and recommendations"""

        print("\n💼 Generating Business Insights and Recommendations")
        print("=" * 60)

        # Best performing model
        best_model_name = max(self.performance_metrics.keys(), 
                             key=lambda x: self.performance_metrics[x]['roc_auc'])
        best_model = self.models[best_model_name]
        best_metrics = self.performance_metrics[best_model_name]

        print(f"🏆 Best Performing Model: {best_model_name}")
        print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
        print(f"   Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall: {best_metrics['recall']:.4f}")

        # Risk threshold analysis
        y_pred_proba = best_model['y_pred_proba']
        y_test = best_model['y_test']

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_analysis = []

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)

            # Calculate metrics at this threshold
            precision = precision_score(y_test, y_pred_thresh, zero_division=0)
            recall = recall_score(y_test, y_pred_thresh, zero_division=0)

            # Business metrics
            total_applicants = len(y_test)
            approved = np.sum(y_pred_thresh == 0)  # Predicted non-defaults
            rejected = np.sum(y_pred_thresh == 1)  # Predicted defaults

            approval_rate = approved / total_applicants
            rejection_rate = rejected / total_applicants

            # False negatives (defaults we missed) - business risk
            false_negatives = np.sum((y_test == 1) & (y_pred_thresh == 0))
            false_negative_rate = false_negatives / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0

            threshold_analysis.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'approval_rate': approval_rate,
                'rejection_rate': rejection_rate,
                'false_negative_rate': false_negative_rate,
                'missed_defaults': false_negatives
            })

        threshold_df = pd.DataFrame(threshold_analysis)

        print("\n📊 Risk Threshold Analysis:")
        print(threshold_df.round(4))

        # Feature importance insights
        if best_model_name in self.feature_importance:
            print(f"\n🔍 Top Risk Factors ({best_model_name}):")
            top_features = self.feature_importance[best_model_name].head(10)
            for idx, row in top_features.iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

        # Business recommendations
        print("\n💡 Business Recommendations:")
        print("=" * 40)

        # Recommended threshold based on business balance
        recommended_threshold = 0.5  # Default
        for _, row in threshold_df.iterrows():
            if row['precision'] >= 0.65 and row['recall'] >= 0.60:  # Good balance
                recommended_threshold = row['threshold']
                break

        recommended_row = threshold_df[threshold_df['threshold'] == recommended_threshold].iloc[0]

        print(f"1. Recommended Risk Threshold: {recommended_threshold}")
        print(f"   - Approval Rate: {recommended_row['approval_rate']:.1%}")
        print(f"   - Precision: {recommended_row['precision']:.1%}")
        print(f"   - Will miss {recommended_row['missed_defaults']} defaults per {len(y_test)} applications")

        print(f"\n2. Conservative Threshold (0.4) for Lower Risk:")
        conservative_row = threshold_df[threshold_df['threshold'] == 0.4].iloc[0]
        print(f"   - Approval Rate: {conservative_row['approval_rate']:.1%}")
        print(f"   - Precision: {conservative_row['precision']:.1%}")
        print(f"   - Will miss {conservative_row['missed_defaults']} defaults")

        print(f"\n3. Aggressive Threshold (0.7) for Higher Volume:")
        aggressive_row = threshold_df[threshold_df['threshold'] == 0.7].iloc[0]
        print(f"   - Approval Rate: {aggressive_row['approval_rate']:.1%}")
        print(f"   - Precision: {aggressive_row['precision']:.1%}")
        print(f"   - Will miss {aggressive_row['missed_defaults']} defaults")

        return threshold_df, recommended_threshold

    def save_models(self, models_dir='models'):
        """Save trained models and results"""

        os.makedirs(models_dir, exist_ok=True)

        print(f"\n💾 Saving models to {models_dir}/...")

        for model_name, model_data in self.models.items():
            # Save model
            model_file = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            model_path = os.path.join(models_dir, model_file)
            joblib.dump(model_data['model'], model_path)
            print(f"✅ Saved {model_name}: {model_path}")

        # Save performance metrics
        metrics_path = os.path.join(models_dir, 'performance_metrics.pkl')
        joblib.dump(self.performance_metrics, metrics_path)

        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(models_dir, 'feature_importance.pkl')
            joblib.dump(self.feature_importance, importance_path)

        print("✅ All models and metrics saved successfully!")

if __name__ == "__main__":
    # This will be called from the main training script
    print("Credit Risk Model Training Module Loaded")
