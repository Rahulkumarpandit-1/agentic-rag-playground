LINEAR REGRESSION

Purpose:
Predict a continuous numerical target.

Examples:
- salary
- house price
- exam score
- temperature

Basic idea:
Find a line that best represents the relationship
between features and a continuous target.

Equation:
y = mx + b

y = prediction
x = feature
m = learned weight/slope
b = learned bias/intercept

Training:
The model learns m and b from training data.

Key idea:
Linear Regression learns a linear relationship
to make numerical predictions.

# Linear Regression — Core Coding Concepts

### `X` and `y`
- **X** → input features used to make a prediction.
- **y** → target value the model needs to predict.

### `train_test_split()`
Splits the dataset into:
- **Training data** → used to teach the model.
- **Test data** → unseen data used to evaluate the model.

### `LinearRegression()`
Creates a Linear Regression model.  
At this point, the model has not learned anything.

### `.fit(X_train, y_train)`
**Trains the model.**  
The model uses the training inputs and their correct targets to learn the relationship between them.

### `.coef_`
The **slope/weight** learned by the model.  
For one feature, it represents how much the prediction changes when that feature increases by 1 unit.

### `.intercept_`
The **starting value/bias** of the learned line when the feature is 0.

### `.predict(X)`
Uses the **trained model** to generate predictions for given input data.

**Remember:**
- `.fit()` → learn
- `.predict()` → predict

### MSE
**Mean Squared Error** measures prediction error by averaging squared errors.

It is useful for measuring how far predictions are from actual values, but its unit is squared.

### RMSE
**Root Mean Squared Error** is the square root of MSE.

It expresses error in the **same unit as the target**, making it easier to understand.

### Evaluation vs Prediction
- **Evaluation:** prediction + known actual target → calculate MSE/RMSE.
- **New prediction:** prediction where the actual target is not known → cannot calculate error yet.

## Linear Regression — Evaluation

MSE (Mean Squared Error):
- Calculates average squared prediction error.
- Formula: average of (actual - prediction)²
- Penalizes larger errors more.
- Unit is squared, so harder to interpret directly.

RMSE (Root Mean Squared Error):
- Square root of MSE.
- Same unit as the target.
- Easier to interpret than MSE.
- Represents typical error magnitude.

Example:
Actual = 32000
Prediction = 32571.43
Error = 571.43

MSE = 326530.61
RMSE = 571.43