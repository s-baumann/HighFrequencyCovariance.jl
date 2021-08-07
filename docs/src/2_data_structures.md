# Data Structures

## Main Structs

`HighFrequencyCovariance` has two main structs. The first is `CovarianceMatrix` which is:

```
mutable struct CovarianceMatrix{R<:Real}
    correlation::Hermitian{R}
    volatility::Vector{R}
    labels::Vector{Symbol}
end
```
A `CovarianceMatrix` struct thus contains three elements. A correlation matrix, a volatility vector and a vector that labels each row/column of the correlation matrix and each row of the volatility vector. Note that an actual covariance matrix is not stored but can be calculated over some interval with the function:
```
covariance(cm::CovarianceMatrix, duration::Real)
```

The second main struct is a `SortedDataFrame` which is:
```
struct SortedDataFrame
    df::DataFrame
    time::Symbol
    grouping::Symbol
    value::Symbol
    groupingrows::Dict{Symbol,Vector{Int64}}
end
```
This presorts a `DataFrame` by time and adds in an index (`groupingrows`) for each asset. Together these allow the covariance estimation functions to run faster. The other struct elements are the labels of the columns of interest in the `DataFrame`.
