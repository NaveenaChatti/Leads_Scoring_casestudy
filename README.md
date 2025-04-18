Lead Conversion Prediction (Logistic Regression)

OBJECTIVE:
Predict which leads are likely to convert using logistic regression.

DATA:
- Source: Leads.csv
- Target: Converted (0/1)

PROCESS:
1. Data Cleaning:
   - Dropped high-missing or imbalanced columns
   - Imputed missing values (mode/median)
   - Combined sparse categories

2. Feature Engineering:
   - Binary mapping (Yes/No → 1/0)
   - Dummy variables for categorical features
   - Outlier treatment on numerical variables

3. EDA:
   - Countplots, barplots, boxplots
   - Correlation heatmap

4. Modeling:
   - Model: Logistic Regression
   - Feature selection using RFE
   - Checked multicollinearity using VIF

5. Evaluation:

   | Dataset | Accuracy | Sensitivity | Specificity | AUC  |
   |---------|----------|-------------|-------------|------|
   | Train   | 92.1%    | 91.5%       | 92.5%       | 0.97 |
   | Test    | 92.6%    | 91.2%       | 93.5%       | 0.97 |

6. Visualization:
   - ROC curve
   - Precision-recall curve
   - Optimal cutoff plot

CONCLUSION:
Model performs well with high accuracy and balanced sensitivity/specificity. Suitable for prioritizing lead follow-ups.

