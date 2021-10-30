"""
    estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts),
                        method::Symbol = :two_scales_volatility;
                        time_grid::Union{Missing,Dict} = missing ,
                        fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                        use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5,
                        num_grids::Real = default_num_grids(ts))

This is a convenience wrapper for the two volatility estimation techniques included in this package.
### General Inputs
* `ts` - The tick data.
* `assets` - What assets from ts that you want to estimate the covariance for.
* `method` - The method can be `:simple_volatility` (for the simple volatility method) or `:two_scales_volatility` (for the two scales volatility method)
#### Inputs only used in `:simple_volatility` method.
* `time_grid` - The grid with which to calculate returns. If missing one is generated with a fixed spacing (if that is provided) or a default spacing.
* `fixed_spacing` - A spacing used to calculate a time grid. Not used if a `time_grid` is input or if `use_all_obs = true`.
* `use_all_obs` - Use all observations to estimate volatilities. Not used if a `time_grid` is provided.
* `rough_guess_number_of_intervals` - A rough number of intervals to calculate a default spacing. Not used if a `time_grid` or `fixed_spacing` is provided or if `use_all_obs = true`.
#### Inputs only used in `:two_scales_volatility` method.
* `num_grids` - Number of grids used in order in two scales estimation.
### Returns
* A `Dict` with estimated volatilities for each asset.
"""
function estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts),
                             method::Symbol = :two_scales_volatility;
                             time_grid::Union{Missing,Dict} = missing ,
                             fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                             use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5,
                             num_grids::Real = default_num_grids(ts))
    if method == :simple_volatility
        return simple_volatility(ts, assets;
                                   time_grid = time_grid, fixed_spacing = fixed_spacing,
                                   use_all_obs = use_all_obs, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :two_scales_volatility
        return two_scales_volatility(ts, assets; num_grids = num_grids)[1]
    else
        error("The volatilty method chosen must be either :simple_volatility or :two_scales_volatility")
    end
end

"""
    estimate_microstructure_noise(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
                                  num_grids::Real = default_num_grids(ts))

This estimates microstructure noise with the two_scales_volatility method.
### Inputs
* `ts` - The tick data.
* `assets` - What assets from ts that you want to estimate the covariance for.
* `num_grids` - Number of grids used in order in two scales estimation.
### Returns
* A `Dict` with estimated microstructure noise variances for each asset.
"""
function estimate_microstructure_noise(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
                             num_grids::Real = default_num_grids(ts))
    return two_scales_volatility(ts, assets; num_grids = num_grids)[2]
end

"""
    estimate_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts),
                        method::Symbol = :preaveraged_covariance;
                        regularisation::Union{Missing,Symbol} = :default, regularisation_params::Dict = Dict(),
                        only_regulise_if_not_PSD::Bool = false,
                        time_grid::Union{Missing,Vector} = missing,
                        fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false,
                        rough_guess_number_of_intervals::Integer = 5,
                        kernel::HFC_Kernel{<:Real} = parzen,
                        H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6,
                        m::Integer = 2, # BNHLS parameters, numJ::Integer = 100, num_blocks::Integer = 10,
                        block_width::Real = (maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])) / num_blocks,
                        microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2],
                        theta::Real = 0.15, g::NamedTuple = g,
                        equalweight::Bool = false, num_grids::Real = default_num_grids(ts))

This is a convenience wrapper for the regularisation techniques.
### General Inputs
* `ts` - The tick data.
* `assets` - What assets from ts that you want to estimate the covariance for.
* `method`  - The method you want to use. This can be `:simple_covariance`, `:bnhls_covariance`, `:spectral_covariance`, `:preaveraged_covariance` or `:two_scales_covariance`.
* `regularisation` - The regularisation method to use. This can be `:identity_regularisation`, `:eigenvalue_clean`, `:nearest_correlation_matrix` or `:nearest_psd_matrix`. You can also choose `:covariance_default` (which is `:nearest_psd_matrix`) or  `:correlation_default` (which is `:nearest_correlation_matrix`). If missing then the default regularisation method for your chosen covariance estimation method will be used.
* `regularisation_params` - Keyword arguments that will be used by your chosen regularisation method.
* `only_regulise_if_not_PSD` - Should the resultant matrix only be regularised if it is not psd.
#### Inputs only used in `:simple_covariance` method.
* `time_grid` - The grid with which to calculate returns (`:simple_covariance` method only).
* `fixed_spacing` - A spacing used to calculate a time grid. Not used if `refresh_times=true` (`:simple_covariance` method only).
* `refresh_times` - Should refresh times be used to estimate covariance (`:simple_covariance` method only).
* `rough_guess_number_of_intervals` - A rough number of intervals to calculate a default spacing. Not used if a `time_grid` or `fixed_spacing` is provided or if `refresh_times=true` (`:simple_covariance` method only).
#### Inputs only used in `:bnhls_covariance` method.
* `kernel` - The kernel used. See the bnhls paper for details. (`:bnhls_covariance` method only)
* `H` - The number of lags/leads used in estimation. See the bnhls paper for details. (`:bnhls_covariance` method only)
* `m` - The number of end returns to average. (`:bnhls_covariance` method only)
#### Inputs only used in `:spectral_covariance` method.
* `numJ` - The number of J values. See the paper for details (`:spectral_covariance` method only).
* `num_blocks` - The number of blocks to split the time frame into. See the preaveraging paper for details (`:spectral_covariance` method only).
* `block_width` - The width of each block to split the time frame into (`:spectral_covariance` method only).
* `microstructure_noise_var` - Estimates of microstructure noise variance for each asset (`:spectral_covariance` method only).
#### Inputs only used in `:preaveraged_covariance` method.
* `drop_assets_if_not_enough_data` - If we do not have enough data to estimate for all the input `assets` should we just calculate the correlation/volatilities for those assets we do have?
* `theta` - A theta value. See paper for details (`:preaveraged_covariance` method only).
* `g` - A tuple containing a preaveraging method (with name "f") and a ψ value. See paper for details (`:preaveraged_covariance` method only).
#### Inputs only used in `:two_scales_covariance` method.
* `equalweight` - Should we use equal weight for the two different linear combinations of assets. If false then an optimal weight is calculated (from volatilities) (`:two_scales_covariance` method only).
* `num_grids` - Number of grids used in order in two scales estimation (`:two_scales_covariance` method only).
### Returns
* A `CovarianceMatrix`
"""
function estimate_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :preaveraged_covariance;
                             regularisation::Union{Missing,Symbol} = :default, regularisation_params::Dict = Dict(),
                             only_regulise_if_not_PSD::Bool = false,
                             time_grid::Union{Missing,Vector} = missing,
                             fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5, # General Inputs
                             kernel::HFC_Kernel{<:Real} = parzen, H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6, m::Integer = 2, # BNHLS parameters
                             numJ::Integer = 100, num_blocks::Integer = 10, block_width::Real = (maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])) / num_blocks, microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2], # Spectral Covariance parameters
                             drop_assets_if_not_enough_data::Bool = false, theta::Real = 0.15, g::NamedTuple = g, # Preaveraging
                             equalweight::Bool = false, num_grids::Real = default_num_grids(ts)) # Two Scales parameters
    if (ismissing(regularisation) == false) && (regularisation == :default)
        regularisation = (method == :two_scales_covariance) ? :correlation_default : :covariance_default
    end

    if method == :simple_covariance
        return simple_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                   time_grid = time_grid,
                                   fixed_spacing = fixed_spacing, refresh_times = refresh_times, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :bnhls_covariance
        return bnhls_covariance(ts, assets; regularisation = regularisation,
                                  only_regulise_if_not_PSD = only_regulise_if_not_PSD, kernel = kernel, H = H,
                                  m = m)
    elseif method == :spectral_covariance
        return spectral_covariance(ts, assets; regularisation = regularisation,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, numJ = numJ, num_blocks = num_blocks, block_width = block_width,
                                     microstructure_noise_var = microstructure_noise_var)
    elseif method == :preaveraged_covariance
        return preaveraged_covariance(ts, assets; regularisation = regularisation, drop_assets_if_not_enough_data = drop_assets_if_not_enough_data,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, theta = theta, g = g)
    elseif method == :two_scales_covariance
        return two_scales_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                    equalweight = equalweight, num_grids = num_grids)
    else
        error("The covariance method chosen must be :simple_covariance, :bnhls_covariance, :spectral_covariance, :preaveraged_covariance or :two_scales_covariance")
    end
end

# The Hermitian version
"""
    regularise(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector, method::Symbol = :correlation_default;
               spacing::Union{Missing,<:Real} = missing,
               weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
               doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)

This is a convenience wrapper for the regularisation techniques.
### General Inputs
* `mat` - The matrix you want to regularise.
* `ts` - The tick data.
* `mat_labels` - The name of the assets for each row/column of the matrix.
* `method`  - The method you want to use. This can be `:identity_regularisation`, `:eigenvalue_clean`, `:nearest_correlation_matrix` or `:nearest_psd_matrix`. You can also choose `:covariance_default` (which is `:nearest_psd_matrix`) or  `:correlation_default` (which is `:nearest_correlation_matrix`).
#### Inputs only used in `:identity_regularisation` method.
* `spacing` - The interval spacing used in choosing an identity weight (`identity_regularisation` method only).
#### Inputs only used in `:nearest_correlation_matrix` method.
* `weighting_matrix` - The weighting matrix used to calculate the nearest psd matrix (`:nearest_correlation_matrix` method only).
* `doDykstra` - Should a Dykstra correction be applied (`:nearest_correlation_matrix` method only).
* `stop_at_first_correlation_matrix` - Should we stop at first valid correlation matrix (`:nearest_correlation_matrix` method only).
* `max_iterates` - Maximum number of iterates (`:nearest_correlation_matrix` method only).
### Returns
* A `Hermitian`


    regularise(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame, method::Symbol = :nearest_correlation_matrix;
               apply_to_covariance::Bool = true,
               spacing::Union{Missing,<:Real} = missing,
               weighting_matrix = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
               doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)

This is a convenience wrapper for the regularisation techniques.
### General Inputs
* `covariance_matrix` - The matrix you want to regularise.
* `ts` - The tick data.
* `method`  - The method you want to use. This can be `:identity_regularisation`, `:eigenvalue_clean`, `:nearest_correlation_matrix` or `:nearest_psd_matrix`. You can also choose `:covariance_default` (which is `:nearest_psd_matrix`) or  `:correlation_default` (which is `:nearest_correlation_matrix`).
* `apply_to_covariance` - Should regularisation be applied to the covariance matrix. If false it is applied to the correlation matrix.
#### Inputs only used in `:identity_regularisation` method.
* `spacing` - The interval spacing used in choosing an identity weight (`identity_regularisation` method only).
#### Inputs only used in `:nearest_correlation_matrix` method.
* `weighting_matrix` - The weighting matrix used to calculate the nearest psd matrix (`:nearest_correlation_matrix` method only).
* `doDykstra` - Should a Dykstra correction be applied (`:nearest_correlation_matrix` method only).
* `stop_at_first_correlation_matrix` - Should we stop at first valid correlation matrix (`:nearest_correlation_matrix` method only).
* `max_iterates` - Maximum number of iterates (`:nearest_correlation_matrix` method only).
### Returns
* A `CovarianceMatrix`
"""
function regularise(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector, method::Symbol = :correlation_default;
                    spacing::Union{Missing,<:Real} = missing,
                    weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
                    doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
    if method == :covariance_default
        method = :nearest_psd_matrix
    elseif method == :correlation_default
        method = :nearest_correlation_matrix
    end
    if method == :identity_regularisation
        return identity_regularisation(mat, ts, mat_labels; spacing = spacing)
    elseif method == :eigenvalue_clean
        return eigenvalue_clean(mat, ts)
    elseif method == :nearest_correlation_matrix
        return nearest_correlation_matrix(mat; weighting_matrix = weighting_matrix, doDykstra = doDykstra,
                                          stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    elseif method == :nearest_psd_matrix
        return nearest_psd_matrix(mat)
    else
        error("The covariance method chosen must be :identity_regularisation, :eigenvalue_clean, :nearest_correlation_matrix or :nearest_psd_matrix. You can also choose :covariance_default (which is :nearest_psd_matrix) or  :correlation_default (which is :nearest_correlation_matrix).")
    end
end
function regularise(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame, method::Symbol = :nearest_correlation_matrix;
                    apply_to_covariance::Bool = true,
                    spacing::Union{Missing,<:Real} = missing,
                    weighting_matrix = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                    doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
    if method == :covariance_default
        method = :nearest_psd_matrix
    elseif method == :correlation_default
        method = :nearest_correlation_matrix
    end
    if method == :identity_regularisation
        return identity_regularisation(covariance_matrix, ts; spacing = spacing, apply_to_covariance = apply_to_covariance)
    elseif method == :eigenvalue_clean
        return eigenvalue_clean(covariance_matrix, ts; apply_to_covariance = apply_to_covariance)
    elseif method == :nearest_correlation_matrix
        return nearest_correlation_matrix(covariance_matrix, ts; weighting_matrix = weighting_matrix, doDykstra = doDykstra,
                                          stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    elseif method == :nearest_psd_matrix
        return nearest_psd_matrix(covariance_matrix; apply_to_covariance = apply_to_covariance)
    else
        error("The covariance method chosen must be :identity_regularisation, :eigenvalue_clean, :nearest_correlation_matrix or :nearest_psd_matrix. You can also choose :covariance_default (which is :nearest_psd_matrix) or  :correlation_default (which is :nearest_correlation_matrix).")
    end
end
