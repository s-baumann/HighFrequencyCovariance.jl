```@meta
CurrentModule = HighFrequencyCovariance
```

# Internal Functions

```@index
Pages = ["internals.md"]
```

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
    g
    get_refresh_times_and_prices
    time_period_ratio
    safe_multiply_period
    weighted_mean
    is_missing_nan_inf
    next_tick
    simple_volatility_given_returns
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
