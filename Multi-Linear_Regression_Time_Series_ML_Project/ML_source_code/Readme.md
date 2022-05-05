# Multiple Linear Regression Time Series

Linear regression is widely used in practice and adapts naturally to even complex forecasting tasks.
The linear regression algorithm learns how to make a weighted sum from its input features. For two features, we would have:
- target = weight_1 * feature_1 + weight_2 * feature_2 + bias

During training, the regression algorithm learns values for the parameters weight_1, weight_2, and bias that best fit the target. (This algorithm is often called ordinary least squares since it chooses values that minimize the squared error between the target and the predictions.) 
The weights are also called regression coefficients and the bias is also called the intercept because it tells you where the graph of this function crosses the y-axis.
Time-step features let you model time dependence. A series is time dependent if its values can be predicted from the time they occurred

## Correlation vs AutoCorrelation

- Correlation is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship. In terms of the strength of relationship, the value of the correlation coefficient varies between +1 and -1.
- A value of ± 1 indicates a perfect degree of association between the two variables. As the correlation coefficient value goes towards 0, the relationship between the two variables will be weaker.
- Auto-correlation refers to the case when your errors are correlated with each other. In layman terms, if the current observation of your dependent variable is correlated with your past observations, you end up in the trap of auto-correlation. 

## Symbolic Regression

Symbolic Regression (SR) is a type of Regression Analysis that searches the space of mathematical expressions to find the model that best fits a given dataset, both in terms of accuracy and simplicity.

## Time Series Basics

-   Chronological Data
- Cannot be shuffled
- Each row indicate specific time record
- Train – Test split happens chronologically
- Data is analyzed univariately (for given use case)
- Nature of the data represents if it can be predicted or not

## Code Description


    File Name : Engine.py
    File Description : Main class for starting different parts and processes of the lifecycle


    File Name : MLR.py
    File Description : Code to train and visualize multiple linear regression with lags


    File Name : PreprocessPlot.py
    File Description : Steps to analyze the visualization


    File Name : SymbolicRegression.py
    File Description : Code to train and visualize SymbolicRegression



## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `MultipleLR.ipynb`

