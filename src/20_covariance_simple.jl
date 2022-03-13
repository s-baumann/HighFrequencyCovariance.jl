
"""
Estimation of the covariance matrix in the standard simple way given returns.
https://en.wikipedia.org/wiki/Sample_mean_and_covariance
"""
function simple_covariance_given_returns(returns::Array{R,2}) where R<:Real
    N = size(returns)[2]
    mat = zeros(N,N)
    for i in 1:N
        for j in i:N
            if i == j
                mat[i,i] = var(returns[:,i])
            else
                mat[i,j] = cov(returns[:,i], returns[:,j])
            end
        end
    end
    return Hermitian(mat)
end

"""
Estimation of the covariance matrix in the standard simple way given a time grid.
"""
function simple_covariance_given_time_grid(ts::SortedDataFrame, assets::Vector{Symbol}, time_grid::Vector; regularisation::Union{Missing,Symbol} = :covariance_default,
                                           regularisation_params::Dict = Dict(), only_regulise_if_not_PSD::Bool = false)
    dd_compiled = latest_value(ts, time_grid; assets = assets)
    dd = get_returns(dd_compiled; rescale_for_duration = false)

    if nrow(dd) < 1 return make_nan_covariance_matrix(assets, ts.time_period_per_unit) end

    returns = Matrix(dd[:, assets])
    covariance = simple_covariance_given_returns(returns)

    # Regularisation
    dont_regulise = ismissing(regularisation) || (only_regulise_if_not_PSD && is_psd_matrix(covariance))
    covariance = dont_regulise ? covariance : regularise(covariance, ts, assets, regularisation; regularisation_params... )

    # av_spacing
    N = length(time_grid)
    spacing = safe_multiply_period(mean(time_grid[2:N] .- time_grid[1:(N-1)]), ts.time_period_per_unit)

    # Packing into a CovarianceMatrix and returning.
    cor, vols = cov2cor_and_vol(covariance, spacing, ts.time_period_per_unit)
    return CovarianceMatrix(cor, vols, assets, ts.time_period_per_unit)
end

"""
    simple_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
                      regularisation::Union{Missing,Symbol} = :covariance_default,
                      regularisation_params::Dict = Dict(), only_regulise_if_not_PSD::Bool = false,
                      time_grid::Union{Missing,Vector} = missing, fixed_spacing::Union{Missing,<:Real} = missing,
                      refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5)

Estimation of the covariance matrix in the standard textbook way.
### Inputs
* `ts` - The tick data.
* `assets` - The assets you want to estimate volatilities for.
* `regularisation` - A symbol representing what regularisation technique should be used. If missing no regularisation is performed.
* `regularisation_params` - keyword arguments to be consumed in the regularisation algorithm.
* `only_regulise_if_not_PSD` - Should regularisation only be attempted if the matrix is not psd already.
* `time_grid` - The grid with which to calculate returns.
* `fixed_spacing` - A spacing used to calculate a time grid. Not used if `refresh_times=true`.
* `refresh_times` - Should refresh times be used to estimate covariance.
* `rough_guess_number_of_intervals` - A rough number of intervals to calculate a default spacing. Not used if a `time_grid` or `fixed_spacing` is provided or if `refresh_times=true`.
### Returns
* A `CovarianceMatrix`.

"""
function simple_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
                           regularisation::Union{Missing,Symbol} = :covariance_default,
                           regularisation_params::Dict = Dict(), only_regulise_if_not_PSD::Bool = false,
                           time_grid::Union{Missing,Vector} = missing, fixed_spacing::Union{Missing,<:Real} = missing,
                           refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5)
    time_grid = get_timegrid(ts, assets, time_grid , fixed_spacing, refresh_times, rough_guess_number_of_intervals)
    return simple_covariance_given_time_grid(ts, assets, time_grid; regularisation = regularisation, regularisation_params = regularisation_params, only_regulise_if_not_PSD = only_regulise_if_not_PSD)
end

"""
    get_timegrid(ts::SortedDataFrame, assets::Vector{Symbol}, time_grid::Missing , fixed_spacing::Union{Missing,<:Real},
                      refresh_times::Bool, rough_guess_number_of_intervals::Integer)
    get_timegrid(ts::SortedDataFrame, assets::Vector{Symbol}, time_grid::Vector, fixed_spacing::Union{Missing,<:Real},
                      refresh_times::Bool, rough_guess_number_of_intervals::Integer)

This returns a sequence of times at which the SortedDataFrame can be queried for prices. This is used in the simple_covariance method.
### Inputs
* `ts` - The tick data.
* `time_grid` - The grid with which to calculate returns.
* `fixed_spacing` - A spacing used to calculate a time grid. Not used if `refresh_times=true`.
* `refresh_times` - Should refresh times be used to estimate covariance.
* `rough_guess_number_of_intervals` - A rough number of intervals to calculate a default spacing. Not used if a `time_grid` or `fixed_spacing` is provided or if `refresh_times=true`.
### Returns
* A `Vector{<:Real}`.
"""
function get_timegrid(ts::SortedDataFrame, assets::Vector{Symbol}, time_grid::Missing , fixed_spacing::Union{Missing,<:Real},
                      refresh_times::Bool, rough_guess_number_of_intervals::Integer)
    time_grid = Vector{eltype(ts.df[:,ts.time])}()
    if !ismissing(fixed_spacing)
        minn, maxx = extrema(ts.df[:,ts.time])
        time_grid = collect(minn:fixed_spacing:maxx)
    elseif refresh_times
        time_grid = get_all_refresh_times(ts, assets)
    else
        n_grid = default_spacing(ts; rough_guess_number_of_intervals = rough_guess_number_of_intervals)
        vals = collect(values(n_grid))
        spacing = min( mean(vals[(isnan.(vals) .== false) .& (isinf.(vals) .== false)]) , duration(ts; in_dates_period = false)/20   )
        spacing = isnan(spacing) ? duration(ts; in_dates_period = false)/20 : spacing
        minn, maxx = extrema(ts.df[:,ts.time])
        time_grid = collect(minn:spacing:maxx)
    end
    return time_grid
end
function get_timegrid(ts::SortedDataFrame,  assets::Vector{Symbol}, time_grid::Vector, fixed_spacing::Union{Missing,<:Real},
                      refresh_times::Bool, rough_guess_number_of_intervals::Integer)
    return time_grid
end
