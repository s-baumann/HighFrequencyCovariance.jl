# Algorithms

## Volatility

There are two built-in algorithms that purely estimate volatility. These are:
* `simple_volatility` estimates the volatility for each stock given a grid of sampling times. If a grid of sampling times is not input then one is estimated optimally (using a formula suggested by Zhang, Mykland & Aït-Sahalia 2005).
* `two_scales_volatility` estimates volatility over two different timescales. One is over a short duration (so a large amount of the measured variation will be from microstructure noise) and one is over a longer duration (so little of the measured variation is from microstructure noise). Then it combines the two estimates to get an estimate of volatility and also of microstructure noise (For how this is done see Zhang, Mykland & Aït-Sahalia 2005).

These two functions can be called directly or through the `estimate_volatility` function. In the `estimate_volatility` function the method can be specified as either `:simple_volatility` or `:two_scales_volatility`.

In addition it is generally possible to infer volatility from the covariance matrix estimates. Given a covariance matrix over some interval it is possible to extract the variance of each stock's price and then determine the volatility from that. In most cases when one of the following covariance estimation function is use it is this measure of volatility that is put into the `CovarianceMatrix` struct.

## Covariance

There are five algorithms for estimating covariances.
* The `simple_covariance` function estimates a covariance matrix in the [basic way](https://en.wikipedia.org/wiki/Sample_mean_and_covariance) using a given timegrid. Users can input their own timegrid, their own interval spacing that will be used to make a timegrid or they can choose to use refresh times (which will likely be biased and so is not recommended). If a user does not do this then by default the spacing will be the average (across assets) optimal spacing that is calculated in the `simple_volatility` function.
* The `preaveraged_covariance` function. This first preaverages price updates over certain intervals and then estimates the correlations using the averaged series. As a result of averaging the microstructure noise is reduced and the correlations are more accurate as a result. As the volatilities of preaveraged returns are artificially low we use two scales volatility estimates for volatility in this case (Christensen, Podolskij and Vetter 2013).
* The `two_scales_covariance` function. This calculates volatilities using the `two_scales_volatility` estimator. It then calculated pairwise correlations by comparing the two scales volatility of different linear combinations of the two assets (Ait-Sahalia, Fan and Xiu 2010).
* The `spectral_covariance` function - The spectral local method of moments technique  (Bibinger, Hautsch, Malec, and Reiss 2014) starts by breaking the trading period into equally sized subintervals. Given each subinterval we  compute a spectral statistic matrix by using a weighted summation of the returns within that interval. We calculate these weights by means of an orthogonal sine function with some spectral frequency j. Then we gather many different spectral statistic matrices by doing this repeatedly with different spectral frequencies. Our estimate of the covariance matrix is then calculated as the average of these spectral statistics.
* The `bnhls_covariance` function - The multivariate realised kernel (Barndorff-Nielsen, Hansen, Lunde, and Shephard 2008 - sometimes called the BNHLS method after the authors) is an algorithm that is designed to provide consistent PSD covariance estimates despite settings where there is microstructure noise (that may not be independent of the underlying price process) and asyncronously traded assets. It is a refinement of an earlier algorithm, the univariate realised kernel estimator, which is faster converging but relies on an assumption of independence between microstructure noise and the underlying price process.

These five functions can be called directly or through the `estimate_covariance` function. In the `estimate_covariance` function the method can be specified as `:simple_covariance`, `:preaveraged_covariance`, `:two_scales_covariance`, `:spectral_covariance` or `:bnhls_covariance`.

## Regularisation

There are four inbuilt regularisation algorithms, `identity_regularisation`, `eigenvalue_clean`, `nearest_psd_matrix` and `nearest_correlation_matrix`. The first three of these can be applied to either the covariance matrix or to the correlation matrix while the fourth can only be applied to the correlation matrix. If input as a regularisation method to a covariance estimation function, these methods can regularise a covariance matrix before it is split into a correlation matrix and volatilities. It can also be applied purely to the resultant correlation matrix. These regularisation techniques can also be applied directly to a `CovarianceMatrix` struct either on the correlation matrix or covariance matrix (in which case a covariance matrix is constructed, regularised and then split into a correlation matrix and volatilities that are then placed in a `CovarianceMatrix` struct).

The regularisation techniques are:
* `identity_regularisation` regularises a covariance (or correlation) by averaging it with an identity matrix of the same dimensions (Ledoit and Wolf 2001).
* `eigenvalue_clean` splits a covariance (or correlation) matrix into its eigenvalue decomposition. Then the distribution of eigenvalues that would be expected if it were a random matrix is computed. Any eigenvalues that are sufficiently small that they could have resulted from a random matrix are averaged together which shrinks their impact while the covariance matrix is still psd (Laloux, Cizeau, Bouchaud and Potters 1999).
* `nearest_psd_matrix` maps an estimated matrix to the nearest PSD (Positive Semi Definite) matrix (Higham 2002).
* `nearest_correlation_matrix` maps an estimated correlation matrix to the nearest PSD matrix. And then the nearest unit diagonal (with off-diagonals less than one in absolute value) matrix. Then the nearest PSD matrix and so on until it converges. The result is the nearest valid correlation matrix.

These four functions can be called directly or through the `regularise` function. In this function the method can be specified as `:identity_regularisation`, `:eigenvalue_clean`, `:nearest_psd_matrix` or `:nearest_correlation_matrix`.
