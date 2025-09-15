
"""
JPMorgan Chase Credit Risk Data Generator
========================================

Generates realistic credit risk datasets with features commonly used
in enterprise credit risk modeling, following JPMorgan Chase standards.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class CreditRiskDataGenerator:
    """Generate comprehensive credit risk datasets"""

    def __init__(self, random_state=42):
        np.random.seed(random_state)
        random.seed(random_state)

    def generate_personal_loan_data(self, n_samples=15000):
        """Generate personal loan dataset with comprehensive credit features"""

        print(f"Generating Personal Loan Credit Dataset: {n_samples} applications")

        # Base demographic and financial data
        data = []

        for i in range(n_samples):
            # Demographics
            age = np.random.normal(42, 12)  # Average age 42
            age = max(18, min(80, int(age)))

            # Employment and Income
            employment_length = max(0, np.random.exponential(8))  # Years employed

            # Income based on age and employment
            base_income = 30000 + age * 800 + employment_length * 1200
            income_noise = np.random.normal(0, 15000)
            annual_income = max(15000, base_income + income_noise)

            # Education level affects income
            education_level = random.choices(
                ['High School', 'Some College', 'Bachelors', 'Masters', 'PhD'],
                weights=[0.25, 0.30, 0.30, 0.12, 0.03]
            )[0]

            education_multiplier = {
                'High School': 0.8, 'Some College': 0.9, 'Bachelors': 1.1, 
                'Masters': 1.3, 'PhD': 1.5
            }
            annual_income *= education_multiplier[education_level]
            annual_income = int(annual_income)

            # Credit History Features
            credit_history_length = min(age - 18, max(0, np.random.exponential(12)))

            # FICO Score (300-850 range)
            base_fico = 650 + (annual_income - 50000) / 2000  # Income correlation
            fico_age_bonus = min(20, age - 25) if age > 25 else 0
            fico_history_bonus = min(30, credit_history_length * 2)

            fico_score = base_fico + fico_age_bonus + fico_history_bonus + np.random.normal(0, 40)
            fico_score = max(300, min(850, int(fico_score)))

            # Debt and Credit Utilization
            total_credit_limit = annual_income * random.uniform(0.3, 2.5)
            credit_utilization = np.random.beta(2, 5)  # Most people use <50% of credit

            # Number of credit accounts
            num_credit_accounts = max(1, int(np.random.poisson(5)))
            num_open_accounts = max(1, int(num_credit_accounts * random.uniform(0.6, 1.0)))

            # Debt calculations
            revolving_balance = total_credit_limit * credit_utilization
            monthly_debt_payments = revolving_balance * 0.03  # ~3% of balance monthly

            # Add installment loans
            installment_debt = annual_income * random.uniform(0, 0.8)
            monthly_debt_payments += installment_debt / 60  # 5-year average term

            # Home ownership
            home_ownership = random.choices(
                ['Rent', 'Own', 'Mortgage', 'Other'],
                weights=[0.35, 0.25, 0.35, 0.05]
            )[0]

            # Loan request details
            loan_amount = random.randint(5000, 50000)
            loan_purpose = random.choices(
                ['debt_consolidation', 'home_improvement', 'major_purchase', 
                 'medical', 'vacation', 'car', 'business', 'other'],
                weights=[0.35, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05]
            )[0]

            # Payment history (number of late payments in last 2 years)
            if fico_score > 740:
                late_payments_2yr = np.random.poisson(0.5)
            elif fico_score > 670:
                late_payments_2yr = np.random.poisson(1.5)
            else:
                late_payments_2yr = np.random.poisson(3.5)

            # Inquiries in last 6 months
            recent_inquiries = max(0, int(np.random.poisson(2)))

            # Employment status
            employment_status = random.choices(
                ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'],
                weights=[0.70, 0.15, 0.12, 0.03]
            )[0]

            # Calculate derived features
            debt_to_income = (monthly_debt_payments * 12) / annual_income
            loan_to_income = loan_amount / annual_income

            # Determine default probability based on risk factors
            risk_score = 0

            # FICO score impact (most important)
            if fico_score < 580:
                risk_score += 0.40
            elif fico_score < 650:
                risk_score += 0.25
            elif fico_score < 720:
                risk_score += 0.10
            else:
                risk_score += 0.02

            # Debt-to-income impact
            if debt_to_income > 0.5:
                risk_score += 0.30
            elif debt_to_income > 0.35:
                risk_score += 0.15
            elif debt_to_income > 0.25:
                risk_score += 0.08

            # Credit utilization impact
            if credit_utilization > 0.9:
                risk_score += 0.20
            elif credit_utilization > 0.7:
                risk_score += 0.12
            elif credit_utilization > 0.3:
                risk_score += 0.05

            # Payment history impact
            risk_score += late_payments_2yr * 0.08

            # Employment impact
            if employment_status == 'Unemployed':
                risk_score += 0.25
            elif employment_status == 'Part-time':
                risk_score += 0.10

            # Add some randomness
            risk_score += random.uniform(-0.1, 0.1)
            risk_score = max(0, min(1, risk_score))

            # Default outcome (1 = default, 0 = no default)
            default = 1 if random.random() < risk_score else 0

            # Loan grade (based on risk assessment)
            if risk_score < 0.05:
                loan_grade = 'A'
            elif risk_score < 0.12:
                loan_grade = 'B'
            elif risk_score < 0.20:
                loan_grade = 'C'
            elif risk_score < 0.30:
                loan_grade = 'D'
            elif risk_score < 0.45:
                loan_grade = 'E'
            else:
                loan_grade = 'F'

            # Interest rate based on grade and market conditions
            base_rates = {'A': 6.5, 'B': 9.2, 'C': 12.8, 'D': 16.5, 'E': 21.2, 'F': 25.8}
            interest_rate = base_rates[loan_grade] + random.uniform(-1.0, 1.5)

            # Create record
            record = {
                'loan_id': f'LC_{i:06d}',
                'loan_amount': loan_amount,
                'loan_purpose': loan_purpose,
                'loan_grade': loan_grade,
                'interest_rate': round(interest_rate, 2),
                'annual_income': annual_income,
                'employment_length': round(employment_length, 1),
                'employment_status': employment_status,
                'home_ownership': home_ownership,
                'fico_score': fico_score,
                'credit_history_length': round(credit_history_length, 1),
                'num_credit_accounts': num_credit_accounts,
                'num_open_accounts': num_open_accounts,
                'total_credit_limit': int(total_credit_limit),
                'revolving_balance': int(revolving_balance),
                'credit_utilization': round(credit_utilization, 3),
                'late_payments_2yr': late_payments_2yr,
                'recent_inquiries': recent_inquiries,
                'monthly_debt_payments': int(monthly_debt_payments),
                'age': age,
                'education_level': education_level,
                'debt_to_income': round(debt_to_income, 3),
                'loan_to_income': round(loan_to_income, 3),
                'risk_score': round(risk_score, 4),
                'default': default
            }

            data.append(record)

        df = pd.DataFrame(data)

        print(f"✅ Generated {len(df)} loan applications")
        print(f"📊 Default rate: {df['default'].mean():.3%}")
        print(f"💰 Average loan amount: ${df['loan_amount'].mean():,.0f}")
        print(f"🏦 Average FICO score: {df['fico_score'].mean():.0f}")

        return df

    def generate_mortgage_data(self, n_samples=8000):
        """Generate mortgage loan dataset"""

        print(f"\nGenerating Mortgage Loan Dataset: {n_samples} applications")

        data = []

        for i in range(n_samples):
            # Property and loan details
            property_value = random.randint(150000, 800000)
            loan_amount = int(property_value * random.uniform(0.6, 0.95))  # LTV ratio
            loan_term = random.choice([15, 20, 30])  # years

            # Borrower profile
            age = max(22, min(70, int(np.random.normal(38, 10))))

            # Income (higher for mortgages)
            annual_income = max(30000, int(np.random.lognormal(np.log(75000), 0.6)))

            # Co-borrower (joint application)
            joint_application = random.choice([0, 1], p=[0.6, 0.4])
            if joint_application:
                co_borrower_income = max(20000, int(np.random.lognormal(np.log(60000), 0.7)))
                total_income = annual_income + co_borrower_income
            else:
                total_income = annual_income
                co_borrower_income = 0

            # FICO scores
            primary_fico = max(300, min(850, int(np.random.normal(720, 60))))
            if joint_application:
                co_borrower_fico = max(300, min(850, int(np.random.normal(710, 65))))
                min_fico = min(primary_fico, co_borrower_fico)
            else:
                co_borrower_fico = 0
                min_fico = primary_fico

            # Debt-to-income calculation
            monthly_income = total_income / 12
            existing_monthly_debt = monthly_income * random.uniform(0.1, 0.4)
            monthly_mortgage_payment = loan_amount / (loan_term * 12) * 1.2  # rough estimate with taxes/insurance

            total_dti = (existing_monthly_debt + monthly_mortgage_payment) / monthly_income

            # Employment
            employment_length = max(0, np.random.exponential(8))
            employment_type = random.choices(
                ['Employed', 'Self-employed', 'Retired', 'Other'],
                weights=[0.80, 0.15, 0.04, 0.01]
            )[0]

            # Property type
            property_type = random.choices(
                ['Single Family', 'Condo', 'Townhouse', 'Multi-family'],
                weights=[0.65, 0.20, 0.12, 0.03]
            )[0]

            # Occupancy
            occupancy = random.choices(
                ['Primary', 'Second Home', 'Investment'],
                weights=[0.80, 0.12, 0.08]
            )[0]

            # Down payment
            down_payment = property_value - loan_amount
            down_payment_percent = down_payment / property_value

            # Loan-to-value ratio
            ltv_ratio = loan_amount / property_value

            # Calculate default probability
            risk_score = 0

            # FICO impact
            if min_fico < 620:
                risk_score += 0.25
            elif min_fico < 680:
                risk_score += 0.12
            elif min_fico < 740:
                risk_score += 0.06
            else:
                risk_score += 0.02

            # DTI impact
            if total_dti > 0.5:
                risk_score += 0.20
            elif total_dti > 0.43:
                risk_score += 0.12
            elif total_dti > 0.36:
                risk_score += 0.06

            # LTV impact
            if ltv_ratio > 0.95:
                risk_score += 0.15
            elif ltv_ratio > 0.90:
                risk_score += 0.08
            elif ltv_ratio > 0.80:
                risk_score += 0.04

            # Employment impact
            if employment_type == 'Self-employed':
                risk_score += 0.08
            elif employment_length < 2:
                risk_score += 0.05

            # Property type impact
            if property_type == 'Multi-family':
                risk_score += 0.05
            elif occupancy == 'Investment':
                risk_score += 0.10

            risk_score += random.uniform(-0.05, 0.05)
            risk_score = max(0, min(1, risk_score))

            default = 1 if random.random() < risk_score else 0

            # Interest rate based on risk
            base_rate = 3.5 + (risk_score * 3)  # 3.5% to 6.5% range
            interest_rate = base_rate + random.uniform(-0.3, 0.3)

            record = {
                'loan_id': f'MTG_{i:06d}',
                'property_value': property_value,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'interest_rate': round(interest_rate, 3),
                'primary_fico': primary_fico,
                'co_borrower_fico': co_borrower_fico if joint_application else None,
                'min_fico': min_fico,
                'annual_income': annual_income,
                'co_borrower_income': co_borrower_income if joint_application else None,
                'total_income': total_income,
                'employment_length': round(employment_length, 1),
                'employment_type': employment_type,
                'property_type': property_type,
                'occupancy': occupancy,
                'ltv_ratio': round(ltv_ratio, 3),
                'down_payment': down_payment,
                'down_payment_percent': round(down_payment_percent, 3),
                'debt_to_income': round(total_dti, 3),
                'monthly_mortgage_payment': int(monthly_mortgage_payment),
                'existing_monthly_debt': int(existing_monthly_debt),
                'age': age,
                'joint_application': joint_application,
                'risk_score': round(risk_score, 4),
                'default': default
            }

            data.append(record)

        df = pd.DataFrame(data)

        print(f"✅ Generated {len(df)} mortgage applications")
        print(f"📊 Default rate: {df['default'].mean():.3%}")
        print(f"🏠 Average property value: ${df['property_value'].mean():,.0f}")
        print(f"💰 Average loan amount: ${df['loan_amount'].mean():,.0f}")

        return df

    def save_datasets(self, data_dir='data'):
        """Generate and save all credit risk datasets"""

        os.makedirs(data_dir, exist_ok=True)

        print("\n🏦 Generating JPMorgan Chase Credit Risk Datasets")
        print("=" * 70)

        # Generate datasets
        personal_loans = self.generate_personal_loan_data()
        mortgage_loans = self.generate_mortgage_data()

        # Save datasets
        personal_path = os.path.join(data_dir, 'personal_loans.csv')
        mortgage_path = os.path.join(data_dir, 'mortgage_loans.csv')

        personal_loans.to_csv(personal_path, index=False)
        mortgage_loans.to_csv(mortgage_path, index=False)

        print(f"\n💾 DATASETS SAVED:")
        print(f"✅ Personal Loans: {personal_path}")
        print(f"✅ Mortgage Loans: {mortgage_path}")

        return {
            'personal_loans': personal_loans,
            'mortgage_loans': mortgage_loans
        }

if __name__ == "__main__":
    generator = CreditRiskDataGenerator()
    datasets = generator.save_datasets()
    print("\n🎉 Credit risk datasets generated successfully!")
