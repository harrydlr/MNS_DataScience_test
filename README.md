# MNS_DataScience_test
This project responds to the Data Science test proposed by M*ns*.

It develops an ARIMA model and a Vector Autoregression (VAR) model to predict the next 30 days expenses of an account.

In the Notebooks folder you can find a notebook in which we briefly explore the data, as well as a notebook in which we study the performance of the VAR model using the MdAPE and MAPE metrics.

The other files correspond to the requested API. The API uses the VAR model (model2.py). In case you want to use the ARIMA model you would have to change "model2" to "model" in the main.py file.

# VAR model Results:
MdAPE:  0.06

MAPE:  136.75

# Conclusions:

The most difficult part of this work has been to think how to deal with the Unevenly Spaced time series problem. I studied the paper: "Algorithms for Unevenly Spaced Time Series: Moving Averages and Other Rolling Operators", Eckner (2017). It gave me an overview of the problem and possible solutions as de EMA(last) method.

The application of the interpolation method using the Scipy package was also explored, but having a lot of time series that were not dense I did not find it appropriate to use it. I implemented the code but it remains to compare results.

In the end, I opted to fill the missing dates with zeros. Assuming that each entry is a snapshot on a specific date, we can be sure that on the missing dates the transactions had a zero amount. The impact and sense of this approach remains to be studied further.

The use of an ARIMA model was studied but it seemed evident that future negative transactions do not depend so much on past values as on the account balance.

The use of Convolutional Neural Networks was also explored but the performance was very poor due to the limited data available for most of the accounts in the dataset.

I therefore decided to implement a Vector Autoregression (VAR) model: 
The data is aggregated weekly, separated by negative and positive transactions, and the account balance.

It seems appropriate to use MAPE to measure the performance of our model. Although it is difficult to interpret for future results, the use of the M(eDian)APE metric gives us a better idea of the performance. As we can see, comparing the MAPE and the MdPE, the model fails for some types of accounts.

# Next steps:
- To compare the results of filling in missing dates using the interpolation method.
- To apply the ExponentialMovingAverage(last) method proposed in the mentioned paper to treat Unevenly Spaced time series.
- To explore exhaustively the appropriate parameters for the model.
