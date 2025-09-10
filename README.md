The goal of this project is to **predict the Chance of Admit** for graduate school applicants based on their academic scores and profile features (GRE, TOEFL, SOP, LOR, CGPA, Research, etc.).

## Dataset
- File: `jamboree.csv`
- Size: 500 records
- Features:
  - GRE Score
  - TOEFL Score
  - University Rating
  - SOP (Statement of Purpose strength)
  - LOR (Letter of Recommendation strength)
  - CGPA
  - Research (0 = No, 1 = Yes)
- Target:
  - Chance of Admit (continuous value between 0 and 1)


##  Models Used
1. **Linear Regression** : a simple, interpretable model that assumes a linear relationship between input features and target.
2. **Random Forest Regressor** : an ensemble of decision trees that can model more complex, non-linear patterns.

## Model Evaluation

Both models were trained on **80% of the dataset** and tested on the remaining **20%**.  
The following metrics were used:


### Results
| Model                  | MSE    | R² Score |
|-------------------------|--------|----------|
| Linear Regression       | 0.0037 | 0.8188   |
| Random Forest Regressor | 0.0043 | 0.7908   |

### Sample Prediction (GRE=350, TOEFL=110, University Rating=5, SOP=4.0, LOR=3.0, CGPA=8.87, Research=0)
- **Linear Regression**: 0.830  
- **Random Forest Regressor**: 0.701  

- **Linear Regression** performed better on this dataset with higher R² and lower MSE.  
- This suggests the relationship between features (like GRE, TOEFL, and CGPA) and the Chance of Admit is mostly linear.  
- **Random Forest** still performed reasonably well, but did not outperform Linear Regression here.  
- For problems with stronger non-linear interactions or categorical data, Random Forest might be more suitable.  
- For this dataset,  Linear Regression is the most suitable model due to its performance and interpretability.
