# Linear Regression
## Description
This program is based on multiple linear regression which the main scope 
is to predict house price based on a set of data such as: `complexAge, totalRooms, totalBedrooms,complexInhabitants, apartmentsNr,  medianComplexValue`.
Input data file is stored in _resoursce_ folder - it is a txt file.

## Data Processing
For computing linear regression and process input data the default tools were used: `pandas , sklearn`.
In order to interpret data, it was analyzed from the graphical stand point of view, using `matplotlib and seaborn libraries`.

![alt text](./resources/heatmap.png)

## Statistic Result
For accuracy measurement is used 3 methods: **Explained Variance Score, RMSE ans R2 Score**.
Trained model has an accuray about 60%
```
{'variance_score': 0.64, 'rmse': 68589.95, 'r2_score': 0.64}
```
![alt text](./resources/score_statistic.png)

## Installation
You can just clone it and run it if you have docker installed on your machine, after navigating to the project folder.
```
1. https://github.com/FilipAdrian/linear-regression.git
2. docker build -t prediction .
3. docker run -p 8080:8080 prediction
```

## CI / CD

Continuous Integration and Continuous Deployment were implemented using Azure Web Apps and GitHub Workflows.
On the GitHub side there is 2 action that are triggered when a commit is pushed to the origin:
- Checking code integrity, which covers syntax error and testing
- Build and Deploy to Azure.

The Azure Web App manage the entire pipeline, it also triggers new commits and at the end deploys the application,
currently it is available on link: `https://price-prediction.azurewebsites.net`. To check if it's available hit 
the endpoint and a message should be displayed: `{"answer": "House Price Prediction"}`, if this massage was returned it 
can be used for predicitons.
In order to identify the possible price for a complex , POST reguest should be made to the: `https://price-prediction.azurewebsites.net/predict` with json payload 
`Json Payload {
    "data":[-122, 37, 52, 8, 707, 1551, 714, 6]
}`, and the response will look like `{
    "answer": 412097.3083
}`
