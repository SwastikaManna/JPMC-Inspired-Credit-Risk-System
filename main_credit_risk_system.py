
"""
JPMorgan Chase Credit Risk Assessment System
==========================================

Main execution script for the complete credit risk modeling pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def setup_environment():
    """Set up the project environment"""

    print("🏦 JPMorgan Chase Credit Risk Assessment System")
    print("=" * 60)
    print("Setting up environment...")

    # Create directories
    directories = ['data', 'models', 'reports', 'logs', 'dashboards']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory: {directory}")

    print("✅ Environment setup complete!")

def run_data_generation():
    """Run data generation pipeline"""

    print("\n📊 Step 1: Generating Credit Risk Dataset")
    print("-" * 40)

    try:
        from credit_data_generator import CreditRiskDataGenerator

        generator = CreditRiskDataGenerator()
        datasets = generator.save_datasets()

        print("✅ Dataset generation completed successfully!")
        return True

    except ImportError:
        print("❌ Error: credit_data_generator.py not found")
        return False
    except Exception as e:
        print(f"❌ Error in data generation: {str(e)}")
        return False

def run_feature_engineering():
    """Run feature engineering pipeline"""

    print("\n🔧 Step 2: Advanced Feature Engineering")
    print("-" * 40)

    try:
        from credit_feature_engineering import CreditFeatureEngineer

        # Load raw data
        data = pd.read_csv('data/personal_loans.csv')

        # Apply feature engineering
        engineer = CreditFeatureEngineer()
        engineered_data = engineer.engineer_features(data)
        X, y = engineer.prepare_for_modeling(engineered_data)

        # Save processed data
        X.to_csv('data/processed_features.csv', index=False)
        y.to_csv('data/processed_target.csv', index=False)

        print("✅ Feature engineering completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        return False

def run_model_training():
    """Run model training and evaluation"""

    print("\n🤖 Step 3: Model Training & Evaluation")
    print("-" * 40)

    try:
        from credit_model_training import CreditRiskModelTrainer

        # Load processed data
        X = pd.read_csv('data/processed_features.csv')
        y = pd.read_csv('data/processed_target.csv').iloc[:, 0]

        # Train models
        trainer = CreditRiskModelTrainer()
        trainer.train_models(X, y)

        # Generate business insights
        threshold_analysis, recommended_threshold = trainer.generate_business_insights()

        # Save models
        trainer.save_models()

        print("✅ Model training completed successfully!")
        print(f"🎯 Recommended threshold: {recommended_threshold}")
        return True

    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return False

def launch_dashboard():
    """Launch the interactive dashboard"""

    print("\n🖥️ Step 4: Launching Interactive Dashboard")
    print("-" * 40)

    try:
        import subprocess
        import sys

        print("Starting Streamlit dashboard...")
        print("Dashboard will be available at: http://localhost:8501")
        print("\nPress Ctrl+C to stop the dashboard")
        print("-" * 40)

        # Launch streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "credit_risk_dashboard.py"])

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {str(e)}")
        print("\nTo manually launch the dashboard, run:")
        print("streamlit run credit_risk_dashboard.py")

def main():
    """Main execution pipeline"""

    print(f"🚀 Starting JPMorgan Chase Credit Risk Assessment Pipeline")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Setup
    setup_environment()

    # Step 2: Data Generation
    if not run_data_generation():
        print("❌ Pipeline stopped due to data generation error")
        return

    # Step 3: Feature Engineering
    if not run_feature_engineering():
        print("❌ Pipeline stopped due to feature engineering error")
        return

    # Step 4: Model Training
    if not run_model_training():
        print("❌ Pipeline stopped due to model training error")
        return

    # Step 5: Success Summary
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ Dataset generated and processed")
    print("✅ Advanced features engineered") 
    print("✅ ML models trained and evaluated")
    print("✅ Business insights generated")
    print("✅ All components ready for deployment")

    print("\n📁 Generated Files:")
    print("  • data/personal_loans.csv - Raw dataset")
    print("  • data/processed_features.csv - Engineered features")
    print("  • models/*.pkl - Trained ML models")
    print("  • reports/executive_business_report.md - Business insights")
    print("  • credit_risk_dashboard.py - Interactive dashboard")

    # Step 6: Dashboard Launch
    user_input = input("\n🖥️ Launch interactive dashboard? (y/n): ").lower().strip()
    if user_input == 'y':
        launch_dashboard()
    else:
        print("\n📝 To launch dashboard later, run:")
        print("streamlit run credit_risk_dashboard.py")

    print("\n🏦 Thank you for using JPMorgan Chase Credit Risk Assessment System!")

if __name__ == "__main__":
    main()
