# LTV Prediction (Classification + Regression Ensemble)

This project focuses on **predicting customer Lifetime Value (LTV)** using a hybrid approach that combines classification and regression models. The solution is designed to work with limited target availability and to better handle high-LTV users.

---

## Project Structure

The project is organized as a sequence of notebooks reflecting the modeling workflow:

1. **Features.ipynb**  
   Feature collection and preprocessing.

2. **Pipeline.ipynb**  
   Model experimentation, cross-validation, metric evaluation, and saving final artifacts.

3. **Final.ipynb**  
   Final model execution without external wrappers, test scoring, and validation of selected features.

---

## Artifacts

All final artifacts are stored in:

artifacts/final/  
├── classificator/   # Classification models with cross-validation  
├── regressor/       # Regression models with cross-validation  
└── test.csv         # Scored test dataset (final_score)

---

## Modeling Approach

The final solution is an **ensemble of two models**:

- **Classifier** — estimates the probability of a user having LTV beyond the first week.  
- **Regressor** — predicts the expected monetary value of LTV.

### Why an Ensemble?

- The observable target **LTV7**, which can be reliably constructed from event data, is only partially aligned with the true LTV target.
- The dataset contains only **608 labeled target cases**, making pure regression unstable.
- The classifier helps estimate continuation behavior, while the regressor focuses on value estimation.

### Regression Models

Several regression setups were explored:

- **Tweedie loss** on the original target (final choice)
- RMSE loss with transformed target:  
  log(target / LTV7 + 1) — tested to improve prediction of high LTV values

Although the transformed target improved metrics for high-LTV users, it degraded performance across other bins, and extreme LTV values were still poorly predicted. As a result, the **Tweedie-based regressor** was selected as the final model.

---

## Final Scoring Logic

The final LTV score is computed as:

final_score = max(LTV7, class_score * regressor_score)

This formulation ensures:
- Consistency with observed early LTV (LTV7)
- Upside potential for users likely to continue monetizing

---

## Metrics

All model metrics and validation results can be found in  
**Pipeline.ipynb → Metrics section**.

---

## Key Observations

One non-obvious behavioral pattern observed during analysis:

If a user is generally active, but their activity decreases toward the end of the week compared to the beginning, the probability of continued monetization after the first week is higher.

---

## Possible Improvements

Planned or potential extensions of the solution:

- Implement greedy feature selection for the final feature set
- Perform more extensive hyperparameter tuning, especially for the regressor
- Train a regression model on **(LTV30 − LTV7)** using a larger dataset to better capture high-LTV users

---

## Notes

This project was built as an end-to-end applied ML pipeline, focusing on:
- Practical target definition under data constraints
- Model stability
- Interpretability of decisions
- Robust scoring logic for production-like scenarios
