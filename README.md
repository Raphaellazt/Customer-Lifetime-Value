## 🚀 Quick Start

Open the notebook directly in Google Colab — no installation required:

[Open in Colab](https://colab.research.google.com/github/Raphaellazt/Customer-Lifetime-Value/blob/main/Customer_Lifetime_Value_.ipynb)

Or run it locally:
```bash
git clone https://github.com/Raphaellazt/Customer-Lifetime-Value.git
cd Customer-Lifetime-Value
pip install pandas numpy scikit-learn matplotlib seaborn scipy plotly fastapi httpx uvicorn joblib
jupyter notebook Customer_Lifetime_Value_.ipynb
```

## 📁 Project Structure
```
Customer-Lifetime-Value/
├── Customer_Lifetime_Value_.ipynb   # Full analysis: data cleaning, feature engineering,
│                                     # model training and evaluation, business scoring,
│                                     # and a tested FastAPI deployment endpoint
└── README.md                        # This file
```

#[README.md](https://github.com/user-attachments/files/32029688/README.md)
# Customer Lifetime Value Predictor

*Predicting customer value to optimize marketing ROI and drive business growth*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview

This project builds a machine learning system that predicts Customer Lifetime Value (CLV) for e-commerce businesses, enabling data-driven marketing decisions and customer segmentation strategies.

**Business Impact:** The selected model (Linear Regression) explains about 7% of CLV variance (R² = 0.069) and identifies 94.5% of genuinely high-value customers (recall), with 58.4% precision when it flags someone as high-value. This is a modest, honest result -- CLV prediction is a hard problem, and this project documents that honestly rather than overstating it.

## 🔍 What I'd Do Differently (And Already Did)

This project didn't start clean. The version I inherited claimed 78% model accuracy and a validated $110K ROI improvement -- numbers that looked great and were completely fabricated: typed into a report template and never replaced with real output. Rather than presenting that version, here's what I actually did:

- **Traced every claimed metric back to code.** Found the real R² was 0.069, not 0.78 -- and rebuilt the entire report around honest numbers.
- **Found and fixed a train/serve skew bug** that silently zeroed out nearly every customer prediction in the deployment demo.
- **Found a second, subtler bug** inside that same fix -- a missing date filter that produced mathematically impossible negative "days since last purchase" values.
- **Corrected a cross-validation data-leakage pattern** that was inflating confidence in the model's stability.
- **Built and tested a real API endpoint**, not just a notebook function -- verified with actual requests, not assumed to work.

The result is a less impressive-looking model (R² = 0.069, not 78%) -- and a portfolio piece that's actually true, with a documented process for catching exactly the kind of error most projects ship with silently.

## 🔧 What It Does

- **Predicts individual customer lifetime value** with roughly $779 average error (MAE) on this dataset
- **Segments customers** into actionable business categories (VIP, High Value, Medium, Low)
- **Optimizes marketing spend** by identifying customers worth targeting
- **Provides business insights** through executive dashboards and ROI analysis

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Model Accuracy (R²) | 0.069 (Linear Regression, selected model) |
| Average Prediction Error (MAE) | $778.95 |
| High-Value Customer Recall | 94.5% |
| High-Value Customer Precision | 58.4% |
| Customer Segmentation Accuracy | 38.6% |
| 5-fold Cross-Validation R² | 0.026 (±0.128) |

*Note: 38.6% segmentation accuracy is across 4 value tiers, where random guessing would score ~25% -- so the model is doing meaningfully better than chance, just not dramatically so.*

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/clv-predictor.git
cd clv-predictor

# Install dependencies
pip install -r requirements.txt

# Run the prediction pipeline
python src/predict_clv.py --input data/new_customers.csv --output results/predictions.csv
```

## 📁 Project Structure
```
clv-predictor/
├── data/
│   ├── raw/                    # Original transaction data
│   ├── processed/              # Cleaned, feature-engineered data
│   └── sample/                 # Sample data for testing
├── src/
│   ├── data_preprocessing.py   # Data cleaning and feature engineering
│   ├── model_training.py       # Model training and evaluation
│   ├── predict_clv.py         # Prediction pipeline
│   └── visualization.py       # Business dashboards and charts
├── models/
│   ├── clv_predictor.pkl      # Trained model
│   ├── feature_scaler.pkl     # Feature preprocessing pipeline
│   └── model_metadata.json    # Model performance metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Initial data analysis
│   ├── 02_feature_engineering.ipynb # Feature creation process
│   ├── 03_model_development.ipynb   # Model training and evaluation
│   └── 04_business_analysis.ipynb   # Business insights and visualization
├── reports/
│   ├── figures/               # Generated charts and visualizations
│   ├── business_report.pdf    # Executive summary and recommendations
│   └── technical_report.md    # Detailed methodology and results
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔍 Methodology

### Data Processing
- **Feature Engineering**: RFM analysis (Recency, Frequency, Monetary) plus behavioral trends
- **Customer Segmentation**: Business-meaningful categories based on predicted value
- **Data Quality**: Systematic handling of missing values, outliers, and data validation

### Machine Learning Approach
- **Algorithm Comparison**: Linear Regression vs. Random Forest vs. Gradient Boosting vs. a small Neural Network -- all four models actually trained in this project
- **Model Selection**: Linear Regression selected on R² -- though the Neural Network posted the lowest MAE of the four, a genuinely interesting result at this sample size
- **Validation**: 5-fold cross-validation (Linear Regression and Random Forest) and holdout testing (all four models)
- **Business Metrics**: Focus on actionable insights over technical metrics

### Key Features
- Customer purchase recency, frequency, and monetary value
- Behavioral trends and spending patterns
- Customer lifecycle stage and tenure
- Purchase pattern analysis and seasonal adjustments

## 📈 Business Applications

### Customer Segmentation
- **VIP Customers** (1000+ predicted CLV): White-glove service, premium offers
- **High Value** (500-1000 predicted CLV): Retention campaigns, loyalty programs
- **Medium Value** (200-500 predicted CLV): Targeted email marketing
- **Low Value** (<200 predicted CLV): Cost-effective acquisition only

### Marketing Optimization
- **Budget Allocation**: Spend 15% of predicted CLV on VIP customers (rule-based, applied in the scoring pipeline)
- **Retention Strategy**: Identify at-risk high-value customers

### ROI Impact (illustrative what-if scenarios, not measured outcomes)
- **Conservative Scenario**: 10% revenue increase, 5% cost savings
- **Realistic Scenario**: 20% revenue increase, 10% cost savings
- **Optimistic Scenario**: 35% revenue increase, 15% cost savings

## 🛠️ Technical Implementation

### Model Performance
```python
# Model comparison results (actual test-set output)
Linear Regression:  R² = 0.069,  MAE = $778.95   ← Selected Model
Random Forest:       R² = 0.049,  MAE = $784.75
Gradient Boosting:   R² = -0.024, MAE = $809.87
Neural Network:       R² = 0.057,  MAE = $765.54   ← lowest MAE of all four
```

### Deployment Pipeline
```python
from src.predict_clv import CLVPredictor

# Load trained model
predictor = CLVPredictor.load('models/clv_predictor.pkl')

# Predict for new customer
clv = predictor.predict_customer(customer_data)
segment = predictor.assign_segment(clv)
```

## 📊 Sample Results

**Customer Scoring Example (real output from this project's scoring pipeline):**
Customer ID: C0757
Predicted CLV: $1241.52
Segment: VIP
Recommended Marketing Budget: $186.23
Strategy: High-touch relationship management

## 🎯 Business Value

This CLV predictor enables:
- **Data-driven marketing decisions** based on customer value predictions
- **Optimized budget allocation** across customer segments
- **Proactive retention strategies** for high-value customers
- **Improved customer acquisition** by identifying valuable customer profiles

## 🔬 Model Validation

- **Cross-validation Score**: 0.026 ± 0.128 (Linear Regression); Random Forest scored −0.065 ± 0.275
- **Statistical Significance**: p = 0.082 -- not a statistically significant difference between the two models
- **Segment Accuracy**: 38.6% of customers correctly classified into 1 of 4 value tiers (vs. ~25% for random guessing)
- **High-Value Recall**: 94.5% of genuinely high-value customers correctly identified

## 🚀 Future Enhancements

- [ ] Real-time prediction API
- [ ] Advanced time-series forecasting
- [ ] Customer churn prediction integration
- [ ] A/B testing framework for model validation
- [ ] Multi-channel attribution modeling
- [ ] Investigate whether a matched (rather than asymmetric) feature/target window recovers more signal

## 📞 Contact

**Raphaella Zuniga** - [raphaellazunigat@gmail.com](mailto:raphaellazunigat@gmail.com)
**LinkedIn**: [linkedin.com/in/raphaella-zuniga](https://linkedin.com/in/raphaella-zuniga)
**Portfolio**: [raphaellaz.com](https://raphaellaz.com)

*This project demonstrates end-to-end machine learning capabilities including data preprocessing, model development, business analysis, and production deployment.*
