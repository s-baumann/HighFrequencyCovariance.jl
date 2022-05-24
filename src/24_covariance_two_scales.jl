"""
    two_scales_correlation(
        prices::DataFrame,
        times::Vector{<:Real},
        asseti::Symbol,
        assetj::Symbol,
        gamma::Real,
        num_grids::Real,
    )

### Inputs
* `prices` - The prices of the stocks.
* `times` - The times corresponding to the prices.
* `asseti` - A symbol for the first asset of the correlation pair you want to calculate for.
* `assetj` -  A symbol for the second asset of the correlation pair you want to calculate for.
* `gamma` - The mixing ratio between the two assets. See paper for details.
* `num_grids` - Number of grids used in order in two scales estimation.
### Returns
* A `Real` for the calculated correlation.
"""
function two_scales_correlation(
    prices::DataFrame,
    times::Vector{<:Real},
    asseti::Symbol,
    assetj::Symbol,
    gamma::Real,
    num_grids::Real,
)
    vol_plus_version = two_scales_volatility(
        prices[:, asseti] * gamma + (1 - gamma) * prices[:, assetj],
        times,
        num_grids,
    )[1]
    vol_minus_version = two_scales_volatility(
        prices[:, asseti] * gamma - (1 - gamma) * prices[:, assetj],
        times,
        num_grids,
    )[1]
    covv = (1 / (4 * gamma * (1 - gamma))) * (vol_plus_version^2 - vol_minus_version^2)
    # Note that we calculate these again rather than feeding in the ones calculated earlier because it is improtant the vols are calculated with the same data as the sums/differences.
    asseti_vol = two_scales_volatility(prices[:, asseti], times, num_grids)[1]
    assetj_vol = two_scales_volatility(prices[:, assetj], times, num_grids)[1]
    correl = covv / (asseti_vol * assetj_vol)
    return correl
end

"""
    get_refresh_times_and_prices(ts::SortedDataFrame, asset1::Symbol, asset2::Symbol)

This returns a vector of prices and refresh times given two input symbols
### Inputs
* `ts` - The tick data.
* `asset1` - The first asset's name.
* `asset2` - The second asset's name.
### Returns
* A `Vector` of prices
* A `Vector` of times corresponding to these prices.
"""
function get_refresh_times_and_prices(ts::SortedDataFrame, asset1::Symbol, asset2::Symbol)
    assets = [asset1, asset2]
    at_times = get_all_refresh_times(ts, assets)
    prices = random_value_in_interval(ts, at_times; assets = assets)
    times = prices[:, :Time]
    return prices, times
end

"""
    two_scales_covariance(
        ts::SortedDataFrame,
        assets::Vector{Symbol} = get_assets(ts);
        regularisation::Union{Missing,Symbol} = :correlation_default,
        regularisation_params::Dict = Dict(),
        only_regulise_if_not_PSD::Bool = false,
        equalweight::Bool = false,
        num_grids::Real = default_num_grids(ts),
        min_obs_for_estimation::Integer = 10,
        if_dont_have_min_obs::Real = NaN,
    )

Estimation of a CovarianceMatrix using the two scale covariance method.
### Inputs
* `ts` - The tick data.
* `assets` - The assets you want to estimate volatilities for.
* `regularisation` - A symbol representing what regularisation technique should be used. If missing no regularisation is performed.
* `regularisation_params` - keyword arguments to be consumed in the regularisation algorithm.
* `only_regulise_if_not_PSD` - Should regularisation only be attempted if the matrix is not psd already.
* `equalweight` - Should we use equal weight for the two different linear combinations of assets. If false then an optimal weight is calculated (from volatilities).
* `num_grids` - Number of grids used in order in two scales estimation.
* `min_obs_for_estimation` - How many observations do we need for estimation. If less than this we use below fallback.
* `if_dont_have_min_obs` - If we do not have sufficient observations to estimate a correlation then what should be used?
### Returns
* A `CovarianceMatrix`.
"""
function two_scales_covariance(
    ts::SortedDataFrame,
    assets::Vector{Symbol} = get_assets(ts);
    regularisation::Union{Missing,Symbol} = :correlation_default,
    regularisation_params::Dict = Dict(),
    only_regulise_if_not_PSD::Bool = false,
    equalweight::Bool = false,
    num_grids::Real = default_num_grids(ts),
    min_obs_for_estimation::Integer = 10,
    if_dont_have_min_obs::Real = NaN,
)
    two_scales_vol, micro_noise = two_scales_volatility(ts, assets; num_grids = num_grids)
    N = length(assets)
    mat = zeros(N, N)
    for i = 1:N
        asseti = assets[i]
        for j = i:N
            assetj = assets[j]
            gamma = min(
                max(
                    0.1,
                    equalweight ? 0.5 :
                        two_scales_vol[assetj] /
                    (two_scales_vol[asseti] + two_scales_vol[assetj]),
                ),
                0.9,
            )
            if (i == j)
                mat[i, j] = 1
            else
                # We do not try to calculate covariances in cases where one asset has zero volatility.
                # covariance is zero in this case but trying to estimate wastes time and can lead
                # to confusing errors for the user.
                zero_vol =
                    (two_scales_vol[asseti] < eps()) || (two_scales_vol[assetj] < eps())
                if zero_vol
                    mat[i, j] = 0.0
                else
                    prices, times = get_refresh_times_and_prices(ts, asseti, assetj)
                    if (nrow(prices) < min_obs_for_estimation)
                        @warn string(
                            "There are only ",
                            nrow(prices),
                            " observations for correlation between ",
                            asseti,
                            " and ",
                            assetj,
                            ". So this correlation will be set as ",
                            if_dont_have_min_obs,
                        )
                        mat[i, j] = if_dont_have_min_obs
                    else
                        mat[i, j] = two_scales_correlation(
                            prices,
                            times,
                            asseti,
                            assetj,
                            gamma,
                            num_grids,
                        )
                    end
                end
            end
        end
    end
    mat = Hermitian(mat)

    # Regularisation - It is done on the correlation matrix for this algo rather than the covariance matrix.
    dont_regulise =
        ismissing(regularisation) || (only_regulise_if_not_PSD && is_psd_matrix(mat))
    covmat = dont_regulise ? mat :
        regularise(mat, ts, assets, regularisation; regularisation_params...)

    vols = map(a -> two_scales_vol[a], assets)
    covmat = CovarianceMatrix(covmat, vols, assets, ts.time_period_per_unit)

    return covmat
end
