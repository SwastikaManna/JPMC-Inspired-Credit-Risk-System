
"""
JPMorgan Chase Credit Risk Feature Engineering
===========================================

Advanced feature engineering for credit risk modeling following
enterprise banking standards and regulatory requirements.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CreditFeatureEngineer:
    """Advanced feature engineering for credit risk models"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def engineer_features(self, df):
        """Apply comprehensive feature engineering"""

        print("🔧 Engineering advanced credit risk features...")

        # Create a copy to avoid modifying original
        data = df.copy()

        # 1. Advanced Debt Ratios
        data['front_end_dti'] = data['monthly_debt_payments'] / (data['annual_income'] / 12)
        data['credit_utilization_squared'] = data['credit_utilization'] ** 2
        data['utilization_times_accounts'] = data['credit_utilization'] * data['num_credit_accounts']

        # 2. FICO Score Buckets and Interactions
        data['fico_excellent'] = (data['fico_score'] >= 800).astype(int)
        data['fico_very_good'] = ((data['fico_score'] >= 740) & (data['fico_score'] < 800)).astype(int)
        data['fico_good'] = ((data['fico_score'] >= 670) & (data['fico_score'] < 740)).astype(int)
        data['fico_fair'] = ((data['fico_score'] >= 580) & (data['fico_score'] < 670)).astype(int)
        data['fico_poor'] = (data['fico_score'] < 580).astype(int)

        # 3. Income-Based Features
        data['income_per_dependent'] = data['annual_income'] / (data.get('dependents', 1) + 1)
        data['loan_amount_to_income'] = data['loan_amount'] / data['annual_income']
        data['income_stability'] = np.where(data['employment_length'] > 5, 1, 0)

        # 4. Credit History Features
        data['credit_age_years'] = data['credit_history_length'] 
        data['avg_account_age'] = data['credit_history_length'] / (data['num_credit_accounts'] + 1)
        data['new_credit_ratio'] = data['recent_inquiries'] / (data['num_credit_accounts'] + 1)

        # 5. Payment Behavior Features
        data['payment_risk'] = np.where(data['late_payments_2yr'] > 2, 1, 0)
        data['payment_score'] = np.clip(100 - (data['late_payments_2yr'] * 15), 0, 100)
        data['recent_credit_seeking'] = np.where(data['recent_inquiries'] > 3, 1, 0)

        # 6. Loan Purpose Risk Categories
        high_risk_purposes = ['debt_consolidation', 'medical', 'business']
        data['high_risk_purpose'] = data['loan_purpose'].isin(high_risk_purposes).astype(int)

        # 7. Employment Risk Factors
        data['employment_risk'] = np.where(
            data['employment_status'].isin(['Unemployed', 'Part-time']), 1, 0
        )
        data['self_employed'] = (data['employment_status'] == 'Self-employed').astype(int)

        # 8. Combined Risk Scores
        data['debt_burden_score'] = (
            data['debt_to_income'] * 0.4 + 
            data['credit_utilization'] * 0.3 + 
            data['loan_to_income'] * 0.3
        )

        data['creditworthiness_score'] = (
            (data['fico_score'] / 850) * 0.5 +
            (data['payment_score'] / 100) * 0.3 +
            (data['income_stability']) * 0.2
        )

        # 9. Categorical Feature Encoding
        categorical_features = ['loan_purpose', 'employment_status', 'home_ownership', 
                              'education_level', 'loan_grade']

        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                data[f'{feature}_encoded'] = le.fit_transform(data[feature].astype(str))
                self.encoders[feature] = le

        # 10. Risk Interaction Features
        data['fico_times_income'] = data['fico_score'] * np.log1p(data['annual_income'])
        data['age_income_interaction'] = data['age'] * np.log1p(data['annual_income'])
        data['credit_mix_score'] = data['num_open_accounts'] / (data['num_credit_accounts'] + 1)

        print(f"✅ Feature engineering completed. Features: {data.shape[1]}")

        return data

    def prepare_for_modeling(self, data, target_column='default'):
        """Prepare data for machine learning models"""

        print("🎯 Preparing data for modeling...")

        # Separate features and target
        X = data.drop([target_column, 'loan_id'], axis=1, errors='ignore')
        y = data[target_column]

        # Select numerical features for scaling
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target-related features that shouldn't be used for prediction
        features_to_remove = ['risk_score']  # This was used to generate target, so remove it
        for feature in features_to_remove:
            if feature in numerical_features:
                numerical_features.remove(feature)
                X = X.drop(feature, axis=1, errors='ignore')

        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = X.copy()

        if numerical_features:
            X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
            self.scalers['features'] = scaler

        print(f"✅ Data prepared for modeling:")
        print(f"   Features: {X_scaled.shape[1]}")
        print(f"   Samples: {X_scaled.shape[0]}")
        print(f"   Default rate: {y.mean():.3%}")

        return X_scaled, y

    def get_feature_importance_names(self, X):
        """Get clean feature names for importance analysis"""

        feature_names = []
        for col in X.columns:
            # Clean up feature names for readability
            clean_name = col.replace('_encoded', '').replace('_', ' ').title()
            feature_names.append(clean_name)

        return feature_names

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/personal_loans.csv')

    # Initialize feature engineer
    engineer = CreditFeatureEngineer()

    # Apply feature engineering
    engineered_data = engineer.engineer_features(data)

    # Prepare for modeling
    X, y = engineer.prepare_for_modeling(engineered_data)

    print(f"\n📊 Final dataset ready for modeling:")
    print(f"   Shape: {X.shape}")
    print(f"   Features: {list(X.columns)[:10]}...")
