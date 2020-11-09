# HighFrequencyCovariance

This package estimates covariance matrix using high frequency data.

This task is more complicated than normal covariance estimation due to several features of high frequency financial data:
* Assets are traded nonsyncronously. For instance we get an updated GOOG price at 14:28:00 and then get an updated price for MSFT only at 14:28:04. Ignoring the asynchronisity with which we get price updates can downwards bias correlations.
* Assets may be traded over different intervals. For instance in calculating the correlation between JP Morgan and Credit Suisse we might have some days where Credit Suisse does  trade in Zurich but due to thanksgiving JP Morgan is not being traded in New York. So we might need to assemble a covariance matrix where the pairwise covariances/correlations come from slightly different intervals.
* Price updates typically contain some "microstructure noise" which reflects frictions in the market rather than the longterm correlations between assets.
* The microstruture noise can often not be iid but can exhibit serial correlation.
* Often more advanced techniques that adjust for the above issues are not guaranteed to return a PSD covariance matrix. So we need to regularise.

HighFrequencyCovariance contains 2 volatility estimators, 5 covariance estimators, 4 regularisation techniques and a number of convenience functions that are intended to overcome these issues and produce reliable correlation, volatility and covariance estimates given high frequency financial data.

A paper briefly outlining each technique is accessible. This paper also contains references to the original econometrics papers for users that seek a more detailed understanding.  The paper also contains a Monte Carlo analysis of the accuracy and time complexity of each algorithm.
This documentation will not replicate all of that content and will instead focus on the practical details on how to write code to use this package.
