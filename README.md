# Air_Passengers_forecast

Problem Statement
An airline company has the data of the number of passengers that have travelled with them on a particular route for the past few years. Using this data, they want to see if they can forecast the number of passengers for the next 12 months.

Making this forecast could be quite beneficial to the company as it would help them take some crucial decisions 

I made this web app through which your select the data for whhich you want to forecast the passengers counts 


Tools Used : Jupyter Notebook, Visual Studio Code
Web APP: Streamlit


Algorithm Used : ARIMA, SARIMA 


Additive Session Decomposition : 


![1](https://user-images.githubusercontent.com/121451346/235794665-84b990a3-d9f1-476d-89d6-a93c04164a80.jpg)

Best p, q, r for ARIMA:
Best (p,d,q) values:  (2, 1, 2)
Best RMSE:  14.002721279540316

Model Prediction Using ARIMA: 

![2](https://user-images.githubusercontent.com/121451346/235794887-a49c0b8f-c89d-4646-8f6d-fb8d598d6db8.jpg)

Best parameters for SARIMA:

Best (P,D,Q,s) values:  (1, 2, 2, 12)
Best RMSE:  14.002721279540316

Model Prediction Using SARIMA: 

![3](https://user-images.githubusercontent.com/121451346/235794948-afa0fa79-1968-4630-8b8e-5e3f5f3a2728.jpg)


Stationarity test : Augmented Dickey-Fuller

p-value: 0.991880
ADF Statistic: 0.8153688792060463
Critical Values: {'1%': -3.4816817173418295, '5%': -2.8840418343195267, '10%': -2.578770059171598}
NON Stationary



