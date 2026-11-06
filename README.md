# Loan-Approval-Prediction
Comparative study of a Multilayer Perceptron, a Deep ANN with early stopping, and a Random Forest for binary loan-approval prediction on 45,000 applicant records. Random Forest achieved the best test accuracy (92.77%).

# overview

Background: Lending institutions must decide quickly and accurately whether to approve a loan. A wrong approval risks default losses, and a wrong rejection loses revenue and denies credit to a creditworthy applicant. Machine learning can capture the non-linear relationships between applicant attributes and repayment behavior that traditional rule-based scoring often misses.

Objective: This project compares two modeling approaches, deep neural networks and tree-based ensembles, on structured credit data. It asks which approach gives the most accurate and interpretable loan-approval predictions, and which applicant attributes matter most.

Data: The public Kaggle loan dataset has 45,000 applicant records with 13 features and no missing values. The features cover demographics, income and employment, loan terms, and credit history. The target is loan_status (approved or rejected). The classes are imbalanced, at about 78% rejected and 22% approved.

Approach: Categorical features were label-encoded and numerical features were standardized, with the scaler fitted on training data only to prevent leakage.
The data was split 80/20 with stratification, giving 36,000 training and 9,000 test samples.
Three models were trained under identical conditions:
a Multilayer Perceptron with dropout
a Deep ANN with early stopping
a Random Forest with 100 trees
Evaluation used accuracy, precision, recall, F1-score and confusion matrices, because accuracy alone can mislead on imbalanced classes.

Results:
Model	Test accuracy
Random Forest	92.77%
Multilayer Perceptron	91.76%
Deep ANN	91.44%

Key findings:
The Random Forest was the most accurate model. It also trained faster than the neural networks and needed no feature scaling.
The strongest predictors were prior loan default history, loan-to-income ratio, interest rate and applicant income. Together they account for over 67% of feature importance.
Gender and education had little influence on predictions (importance below 0.02 each).

Conclusion and future work: For structured tabular credit data, tree-based ensembles can match or beat deep neural networks while being cheaper to train and easier to interpret. Planned extensions are gradient-boosted models (XGBoost, LightGBM), hyperparameter tuning, class-imbalance handling (SMOTE, class weights), and SHAP-based interpretability.
