function two_scales_correlation(prices::DataFrame, times::Vector{<:Real}, asseti::Symbol, assetj::Symbol, gamma::Real, num_grids::Real)
    vol_plus_version  = two_scales_volatility(prices[:,asseti] * gamma + (1-gamma) * prices[:,assetj] , times, Symbol("CompoundAsset__",asseti,"__+__",assetj,"__",gamma), num_grids)[1]
    vol_minus_version = two_scales_volatility(prices[:,asseti] * gamma - (1-gamma) * prices[:,assetj] , times, Symbol("CompoundAsset__",asseti,"__-__",assetj,"__",gamma), num_grids)[1]
    covv = (1/(4*gamma*(1-gamma))) * ( vol_plus_version^2 -  vol_minus_version^2 )
    # Note that we calculate these again rather than feeding in the ones calculated earlier because it is improtant the vols are calculated with the same data as the sums/differences.
    asseti_vol = two_scales_volatility(prices[:,asseti], times, Symbol("CompoundAsset__",asseti,"__ONLY"), num_grids)[1]
    assetj_vol = two_scales_volatility(prices[:,assetj], times, Symbol("CompoundAsset__",assetj,"__ONLY"), num_grids)[1]
    correl = covv/(asseti_vol * assetj_vol)
    return correl
end

function get_refresh_times_and_prices(ts::SortedDataFrame, asset1::Symbol, asset2::Symbol)
    assets = [asset1, asset2]
    at_times    = get_all_refresh_times(ts, assets)
    prices = random_value_in_interval(ts, at_times; assets = assets)
    times = prices[:,:Time]
    return prices, times
end

"""
Estimation of a CovarianceMatrix using the two scale covariance method.
"""
function two_scales_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);  regularisation::Union{Missing,Symbol} = :CorrelationDefault, regularisation_params::Dict = Dict(),
                             only_regulise_if_not_PSD::Bool = false, equalweight::Bool = false, num_grids::Real = default_num_grids(ts))

    two_scales_vol, micro_noise = two_scales_volatility(ts, assets; num_grids = num_grids)

    N = length(assets)
    mat = zeros(N,N)
    for i in 1:N
        asseti = assets[i]
        for j in i:N
            assetj = assets[j]
            gamma = min(max(0.1, equalweight ? 0.5 : two_scales_vol[assetj] / (two_scales_vol[asseti] + two_scales_vol[assetj]))   , 0.9)
            if (i == j)
                mat[i,j] = 1
            else
                zero_vol = (two_scales_vol[asseti] < 2*eps()) |  (two_scales_vol[assetj] < 2*eps())
                if zero_vol
                    mat[i,j] = 0.0
                else
                    prices, times = get_refresh_times_and_prices(ts, asseti, assetj)
                    mat[i,j] = two_scales_correlation(prices, times, asseti, assetj, gamma, num_grids)
                end
            end
        end
    end
    mat = Hermitian(mat)

    # Regularisation - It is done on the correlation matrix for this algo rather than the covariance matrix.
    dont_regulise = ismissing(regularisation) || (only_regulise_if_not_PSD && is_psd_matrix(mat))
    covmat = dont_regulise ? mat : regularise(mat, ts, assets, regularisation; regularisation_params... )

    vols = map(a -> two_scales_vol[a], assets)
    covmat = CovarianceMatrix(covmat, vols, assets)

    return covmat
end
