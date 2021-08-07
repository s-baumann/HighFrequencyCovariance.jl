
```@meta
CurrentModule = HighFrequencyCovariance
```

# Structs

```@index
Pages = ["types.md"]
```

The `SortedDataFrame` struct wraps a `DataFrame` (from the `DataFrames` package). In the constructor function for the `DataFrame`
we presort the data and create a mapping `Dict` so that it is fast to subset the
`DataFrame` by the group.

The `CovarianceMatrix` mutable struct stores three elements. A `Hermitian` correlation matrix, a vector of volatilities
and a vector of labels. The order of the labels matches the order of the assets in
the volatility vector and correlation matrix.

## Structs specification

```@docs
SortedDataFrame
CovarianceMatrix
```
