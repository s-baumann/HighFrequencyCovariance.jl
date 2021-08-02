```@meta
CurrentModule = HighFrequencyCovariance
```

# Estimation Functions

```@index
Pages = ["functions.md"]
```

## Estimating Volatility

The `estimate_volatility` function is the main volatility estimation function. Either of the two estimation methods can be called by specifying `:simple_volatility` or `:two_scales_volatility` as the method argument in the `estimate_volatility` function. Alternatively the `simple_volatility` or `two_scales_volatility` functions can be called directly.

The `simple_volatility` returns a  `Dict` with the estimated volatility for each asset. The `two_scales_volatility` function on the other hand returns a tuple with a `Dict` of estimated volatilities in the first position and a `Dict` of estimated microstructure noise variances in the second. For uniformity of output the `estimate_volatility` returns a `Dict` with the estimated volatility for each asset regardless of what method is chosen.

If a user wants to calculate both volatilities and microstructure noises then they are advised to prefer the `two_scales_volatility` function over doing both `estimate_volatility` (with the `:two_scale_covariance` method argument) and the `estimate_microstructure_noise` function. While the results are the same doing the two function option means everything is calculated twice.

```@docs
    estimate_volatility
    simple_volatility
    two_scales_volatility
```

## Estimating Microstructure Noise

There is one function that returns a `Dict` of microstructure noise estimates for each asset. These estimates come from the `two_scales_volatility` method and are identical to what you get if you examine the second element of the tuple that that function outputs.
```@docs
    estimate_microstructure_noise
```

## Estimating Covariance Matrices

The `estimate_covariance` is the main method for estimating a CovarianceMatrix. Five possible methods can be input to this function (or the functions for each method can alternatively be called directly).

All covariance estimation functions take in a `SortedDataFrame` and (optionally) a vector of symbol names representing assets and (optionally) a specified regularisation method. If the vector of symbol names for assets in input then the CovarianceMatrix will only include those input assets and assets will be in the order specified in the vector.

If the regularisation method is specified then this will be used in regularising the resulting matrix. This can alternatively be `missing` in which case no regularisation will be done. By default the `nearest_psd_matrix` will be used for every method except the `two\_scales\_covariance` method and this regularisation is done on the estimated covariance matrix before its correlation matrix and volatilities are split up and placed in a `CovarianceMatrix` struct. For the `two\_scales\_covariance` method the correlation matrix is estimated directly and regularisation is applied to this correlation matrix. Hence the `nearest_correlation_matrix` is the default.

Note that some combinations of estimation technique and regularisation technique will not work. For instance `nearest_correlation_matrix` would not be good to apply in the case of the `preaveraged_covariance` method as it would attempt to make a covariance matrix into a correlation matrix with a unit diagonal. In addition if the estimated matrix is very non-psd then heavy regularisation might be required. This may have bad results. In these cases it may be useful to turn off regularisation in the estimation function and instead apply regularisation to the `CovarianceMatrix` struct.

```@docs
      estimate_covariance
      simple_covariance
      bnhls_covariance
      spectral_covariance
      preaveraged_covariance
      two_scales_covariance
```

## Regularisation of Covariance Matrices

The main function for regularisation is the `regularise` function. In addition four methods are implemented for regularising matrices can be used directly or through the `regularise` function. All of these functions can be applied to either a `Hermitian` matrix or to a `CovarianceMatrix` struct.

If these functions are applied to a `Hermitian` then regularisation is applied and a regularised `Hermitian` is returned.

If these functions are applied to a `CovarianceMatrix` struct.

```@docs
    regularise
    identity_regularisation
    eigenvalue_clean
    nearest_psd_matrix
    nearest_correlation_matrix
```
