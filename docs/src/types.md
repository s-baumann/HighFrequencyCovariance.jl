
```@meta
CurrentModule = HighFrequencyCovariance
```

# Types

```@index
Pages = ["types.md"]
```

## Type hierarchy design

`SortedDataFrame` struct wraps a DataFrame (from the DataFrames package). In the constructor function for the DataFrame
we presort the data and create a mapping dict so that it is fast to subset the
DataFrame by the group.

`CovarianceMatrix`. This stores three elements. A Hermitian correlation matrix, a vector of volatilities
and a vector of labels. The order of the labels matches the order of the assets in
the volatility vector and correlation matrix.

## Types specification

```@docs
SortedDataFrame
CovarianceMatrix
```
