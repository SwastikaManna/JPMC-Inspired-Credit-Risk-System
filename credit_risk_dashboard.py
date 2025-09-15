
"""
JPMorgan Chase Credit Risk Assessment Dashboard
==============================================

Interactive dashboard for credit risk scoring and business decision making
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="JPMorgan Chase Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #0f4c75;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    border-bottom: 3px solid #0f4c75;
    background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
}

.metric-card {
    background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.risk-high {
    background-color: #dc3545;
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    text-align: center;
    font-weight: bold;
}

.risk-medium {
    background-color: #fd7e14;
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    text-align: center;
    font-weight: bold;
}

.risk-low {
    background-color: #198754;
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    text-align: center;
    font-weight: bold;
}

.business-insight {
    background-color: #f8f9fa;
    border-left: 4px solid #0f4c75;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load credit risk data and models"""
    try:
        # Load processed data
        X = pd.read_csv('data/processed_features.csv')
        y = pd.read_csv('data/processed_target.csv')
        raw_data = pd.read_csv('data/personal_loans.csv')

        return X, y.iloc[:, 0], raw_data  # y is a single column DataFrame
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        models['logistic'] = joblib.load('models/logistic_regression_model.pkl')
        models['random_forest'] = joblib.load('models/random_forest_model.pkl')
        models['metrics'] = joblib.load('models/performance_metrics.pkl')
        models['feature_importance'] = joblib.load('models/feature_importance.pkl')
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")

    return models

def create_risk_assessment_interface(models, X, raw_data):
    """Create individual risk assessment interface"""

    st.subheader("🎯 Individual Loan Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Borrower Information**")
        fico_score = st.slider("FICO Score", 300, 850, 720)
        annual_income = st.number_input("Annual Income ($)", 15000, 200000, 50000)
        employment_length = st.slider("Employment Length (years)", 0.0, 25.0, 5.0)
        age = st.slider("Age", 18, 80, 35)

    with col2:
        st.markdown("**Loan Details**")
        loan_amount = st.number_input("Loan Amount ($)", 1000, 50000, 15000)
        loan_purpose = st.selectbox("Loan Purpose", 
            ['debt_consolidation', 'home_improvement', 'major_purchase', 
             'medical', 'vacation', 'car', 'business', 'other'])
        home_ownership = st.selectbox("Home Ownership", 
            ['Rent', 'Own', 'Mortgage', 'Other'])

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Credit History**")
        credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3, 0.01)
        late_payments_2yr = st.number_input("Late Payments (2 years)", 0, 10, 1)
        num_credit_accounts = st.slider("Number of Credit Accounts", 1, 20, 6)

    with col4:
        st.markdown("**Additional Factors**")
        employment_status = st.selectbox("Employment Status", 
            ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
        recent_inquiries = st.slider("Recent Credit Inquiries", 0, 10, 2)
        monthly_debt = st.number_input("Monthly Debt Payments ($)", 0, 10000, 500)

    if st.button("🔍 Assess Risk", type="primary"):
        # Create a sample for prediction (simplified)
        # In a real implementation, you'd need to properly engineer features
        sample_features = create_sample_features(
            fico_score, annual_income, employment_length, age, loan_amount,
            credit_utilization, late_payments_2yr, num_credit_accounts, monthly_debt
        )

        # Get prediction from best model (logistic regression)
        if 'logistic' in models:
            try:
                # Use a simplified approach - create a feature vector with same length as training
                # This is a demo version - in production you'd need proper feature engineering
                sample_array = np.zeros(X.shape[1])

                # Map some key features (this is simplified for demo)
                if 'fico_score' in X.columns:
                    sample_array[X.columns.get_loc('fico_score')] = fico_score
                if 'annual_income' in X.columns:
                    sample_array[X.columns.get_loc('annual_income')] = annual_income
                if 'loan_amount' in X.columns:
                    sample_array[X.columns.get_loc('loan_amount')] = loan_amount

                risk_probability = models['logistic'].predict_proba([sample_array])[0][1]

                # Display risk assessment
                col_left, col_right = st.columns(2)

                with col_left:
                    st.metric("Default Risk Probability", f"{risk_probability:.1%}")

                with col_right:
                    if risk_probability >= 0.7:
                        st.markdown('<div class="risk-high">🔴 HIGH RISK</div>', unsafe_allow_html=True)
                        recommendation = "DECLINE - High probability of default"
                    elif risk_probability >= 0.4:
                        st.markdown('<div class="risk-medium">🟡 MEDIUM RISK</div>', unsafe_allow_html=True)
                        recommendation = "REVIEW REQUIRED - Additional verification needed"
                    else:
                        st.markdown('<div class="risk-low">🟢 LOW RISK</div>', unsafe_allow_html=True)
                        recommendation = "APPROVE - Low default risk"

                st.markdown(f"**Recommendation:** {recommendation}")

                # Show key risk factors
                st.markdown("**Key Risk Factors:**")
                factors = []
                if fico_score < 650:
                    factors.append("⚠️ FICO score below 650")
                if credit_utilization > 0.7:
                    factors.append("⚠️ High credit utilization")
                if late_payments_2yr > 2:
                    factors.append("⚠️ Multiple recent late payments")
                if loan_amount / annual_income > 0.5:
                    factors.append("⚠️ High loan-to-income ratio")

                if factors:
                    for factor in factors:
                        st.write(factor)
                else:
                    st.success("✅ No major risk factors identified")

            except Exception as e:
                st.error(f"Error in risk assessment: {str(e)}")

def create_sample_features(fico, income, emp_length, age, loan_amt, util, late_pays, accounts, monthly_debt):
    """Create sample feature vector for prediction"""
    # This is a simplified version for demo purposes
    return {
        'fico_score': fico,
        'annual_income': income,
        'employment_length': emp_length,
        'age': age,
        'loan_amount': loan_amt,
        'credit_utilization': util,
        'late_payments_2yr': late_pays,
        'num_credit_accounts': accounts,
        'monthly_debt_payments': monthly_debt,
        'debt_to_income': (monthly_debt * 12) / income,
        'loan_to_income': loan_amt / income
    }

def create_business_analytics(models, raw_data, y):
    """Create business analytics and insights"""

    st.subheader("📊 Business Analytics & Insights")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Applications", f"{len(raw_data):,}")

    with col2:
        st.metric("Default Rate", f"{y.mean():.1%}")

    with col3:
        st.metric("Avg FICO Score", f"{raw_data['fico_score'].mean():.0f}")

    with col4:
        st.metric("Avg Loan Amount", f"${raw_data['loan_amount'].mean():,.0f}")

    # Performance metrics
    if 'metrics' in models:
        st.markdown("### 🏆 Model Performance")

        metrics_data = []
        for model_name, metrics in models['metrics'].items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.1%}",
                'Precision': f"{metrics['precision']:.1%}",
                'Recall': f"{metrics['recall']:.1%}",
                'ROC-AUC': f"{metrics['roc_auc']:.3f}"
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    # Feature importance
    if 'feature_importance' in models and 'Random Forest' in models['feature_importance']:
        st.markdown("### 🔍 Top Risk Factors")

        top_features = models['feature_importance']['Random Forest'].head(8)

        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='#0f4c75'
        ))

        fig.update_layout(
            title="Most Important Risk Factors",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Risk distribution by FICO
    st.markdown("### 📈 Risk Analysis by FICO Score")

    # Create FICO bins
    bins = [300, 580, 670, 740, 800, 850]
    labels = ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Very Good (740-799)', 'Excellent (800+)']

    raw_data['fico_bin'] = pd.cut(raw_data['fico_score'], bins=bins, labels=labels, right=False)

    fico_analysis = raw_data.groupby('fico_bin').agg({
        'default': ['count', 'mean']
    }).round(3)

    fico_analysis.columns = ['Count', 'Default Rate']
    fico_analysis['Default Rate %'] = (fico_analysis['Default Rate'] * 100).round(1)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Application Volume by FICO', 'Default Rate by FICO'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    fig.add_trace(
        go.Bar(x=fico_analysis.index, y=fico_analysis['Count'], name="Applications"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=fico_analysis.index, y=fico_analysis['Default Rate %'], name="Default Rate %"),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def create_threshold_analysis():
    """Create risk threshold analysis for business decisions"""

    st.subheader("⚖️ Risk Threshold Optimization")

    st.markdown("""
    <div class="business-insight">
    <strong>Business Impact of Risk Thresholds</strong><br>
    Adjust the risk threshold to balance approval rates with default risk. 
    Lower thresholds = more approvals but higher risk. Higher thresholds = fewer approvals but lower risk.
    </div>
    """, unsafe_allow_html=True)

    # Threshold analysis data from our model results
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    approval_rates = [15.9, 29.6, 43.5, 52.2, 67.2, 76.9, 93.9]
    precision_rates = [41.4, 45.7, 50.6, 53.9, 61.1, 66.4, 81.0]

    threshold_df = pd.DataFrame({
        'Threshold': thresholds,
        'Approval Rate (%)': approval_rates,
        'Precision (%)': precision_rates
    })

    # Interactive threshold selector
    selected_threshold = st.slider("Risk Threshold", 0.2, 0.8, 0.4, 0.1)

    # Find closest threshold in our data
    closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - selected_threshold))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Approval Rate", f"{approval_rates[closest_idx]:.1f}%")

    with col2:
        st.metric("Precision", f"{precision_rates[closest_idx]:.1f}%")

    with col3:
        estimated_approvals = int(approval_rates[closest_idx] / 100 * 1000)  # Per 1000 applications
        st.metric("Approvals per 1000", f"{estimated_approvals}")

    # Visualization
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds, y=approval_rates,
        mode='lines+markers',
        name='Approval Rate (%)',
        line=dict(color='#3282b8', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=thresholds, y=precision_rates,
        mode='lines+markers', 
        name='Precision (%)',
        line=dict(color='#0f4c75', width=3),
        marker=dict(size=8)
    ))

    # Highlight selected threshold
    fig.add_vline(
        x=selected_threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Selected: {selected_threshold}"
    )

    fig.update_layout(
        title="Risk Threshold Impact on Business Metrics",
        xaxis_title="Risk Threshold",
        yaxis_title="Percentage (%)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown("### 💡 Business Recommendations")

    if selected_threshold <= 0.3:
        st.info("🔵 **Conservative Strategy**: Low approval rate but high precision. Suitable for risk-averse periods.")
    elif selected_threshold <= 0.5:
        st.success("🟢 **Balanced Strategy**: Moderate approval rate with good precision. Recommended for normal operations.")
    else:
        st.warning("🟡 **Growth Strategy**: High approval rate but lower precision. Suitable for market expansion.")

def main():
    """Main dashboard function"""

    # Header
    st.markdown('<h1 class="main-header">🏦 JPMorgan Chase Credit Risk Assessment System</h1>', unsafe_allow_html=True)

    # Load data and models
    X, y, raw_data = load_data()
    models = load_models()

    if X is None or y is None or not models:
        st.error("Failed to load required data or models. Please check your files.")
        st.stop()

    # Sidebar navigation
    st.sidebar.title("🗂️ Navigation")

    page = st.sidebar.radio(
        "Select Analysis Type:",
        ["📊 Business Dashboard", "🎯 Individual Assessment", "⚖️ Threshold Analysis", "📈 Portfolio Analytics"]
    )

    # Main content based on selection
    if page == "📊 Business Dashboard":
        create_business_analytics(models, raw_data, y)

    elif page == "🎯 Individual Assessment":
        create_risk_assessment_interface(models, X, raw_data)

    elif page == "⚖️ Threshold Analysis":
        create_threshold_analysis()

    elif page == "📈 Portfolio Analytics":
        st.subheader("📈 Portfolio Risk Analytics")

        # Portfolio summary
        col1, col2, col3 = st.columns(3)

        with col1:
            total_exposure = raw_data['loan_amount'].sum()
            st.metric("Total Loan Exposure", f"${total_exposure:,.0f}")

        with col2:
            avg_risk = y.mean()
            potential_losses = total_exposure * avg_risk * 0.6  # Assuming 60% loss given default
            st.metric("Estimated Potential Losses", f"${potential_losses:,.0f}")

        with col3:
            high_risk_loans = len(raw_data[raw_data['fico_score'] < 650])
            st.metric("High Risk Applications", f"{high_risk_loans:,}")

        # Risk distribution
        fig = px.histogram(
            raw_data, x='fico_score', nbins=30,
            title="FICO Score Distribution",
            color_discrete_sequence=['#0f4c75']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Loan amount vs default risk
        fig2 = px.scatter(
            raw_data, x='loan_amount', y='fico_score', 
            color='default', size='annual_income',
            title="Loan Amount vs FICO Score (colored by default status)",
            color_discrete_map={0: '#198754', 1: '#dc3545'}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**JPMorgan Chase Credit Risk Assessment System** | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        "Built with Streamlit & Machine Learning"
    )

if __name__ == "__main__":
    main()
