
```@meta
CurrentModule = HighFrequencyCovariance
```

# Helper Functions

```@index
Pages = ["helper_functions.md"]
```

## Working with SortedDataFrame structs

```@index
    get_assets
    ticks_per_asset
    duration
    subset_to_tick
    subset_to_time
```

## Working with CovarianceMatrix structs

```@index
    covariance
    get_correlation
    get_volatility
    make_nan_covariance_matrix
    combine_covariance_matrices
    rearrange
    cov2cor
    cor2cov
    cov2cor_and_vol
    construct_matrix_from_eigen
    get_returns
    valid_correlation_matrix
    is_psd_matrix
```

## Blocking and Regularisation Functions

```@docs
    put_assets_into_blocks_by_trading_frequency
    blockwise_estimation
```

## Monte Carlo

```@docs
    generate_random_path
    ItoSet
```

## For getting a DataFrame version of a CovarianceMatrix and vice versa.

```@docs
    to_dataframe
    dataframe_to_covariancematrix
```

## Blocking and Regularisation Functions

```@docs
    put_assets_into_blocks_by_trading_frequency
    blockwise_estimation
```

## Internal Functions potentially of use for advanced users

### Metrics for distances between CovarianceMatrix structs

```@docs
    calculate_mean_abs_distance
    squared_frobenius
    squared_frobenius_distance
```

### Used in volatility estimation techniques

```@docs
    default_num_grids
```

### Used in covariance estimation techniques

```@docs
    get_all_refresh_times
    latest_value
    time_between_refreshes
    random_value_in_interval
```

### Kernels used in the BNHLS method

```@docs
    HFC_Kernel
    parzen
    quadratic_spectral
    fejer
    tukey_hanning
    bnhls_2008
```

### Used in nearest correlation regularisation
```@docs
    project_to_S
    project_to_U
    iterate_higham
```
