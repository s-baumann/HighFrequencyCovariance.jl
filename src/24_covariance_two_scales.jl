function two_scales_correlation(prices::DataFrame, times::Vector{<:Real}, asset1::Symbol, asset2::Symbol, purevols::Dict{Symbol,<:Real}, gamma::Real , num_grids::Real, return_calc::Function)
    vol_plus_version  = two_scales_volatility(prices[:,asset1] * gamma + (1-gamma) * prices[:,asset2] , times, Symbol("CompoundAsset__",asset1,"__+__",asset2,"__",gamma), num_grids, return_calc)[1]
    vol_minus_version = two_scales_volatility(prices[:,asset1] * gamma - (1-gamma) * prices[:,asset2] , times, Symbol("CompoundAsset__",asset1,"__-__",asset2,"__",gamma), num_grids, return_calc)[1]
    covv = (1/(4*gamma*(1-gamma))) * ( vol_plus_version^2 -  vol_minus_version^2 )
    correl = covv/(purevols[asset1] * purevols[asset2])
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
function two_scales_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = nearest_correlation_matrix,
                             only_regulise_if_not_PSD::Bool = false, equalweight::Bool = false, num_grids::Real = 10, return_calc::Function = simple_differencing)

    two_scales_vol, micro_noise = two_scales_volatility(ts, assets)

    N = length(assets)
    mat = zeros(N,N)
    for i in 1:N
        asseti = assets[i]
        for j in i:N
            assetj = assets[j]
            gamma = equalweight ? 0.5 : two_scales_vol[assetj] / (two_scales_vol[asseti] + two_scales_vol[assetj])
            if (i == j)
                mat[i,j] = 1
            else
                prices, times = get_refresh_times_and_prices(ts, asseti, assetj)
                mat[i,j] = two_scales_correlation(prices, times, asseti, assetj, two_scales_vol, gamma, num_grids, return_calc)
            end
        end
    end
    mat = Hermitian(mat)
    vols = map(a -> two_scales_vol[a], assets)
    covmat = CovarianceMatrix(Hermitian(mat), vols, assets)

    # Regularisation
    dont_regulise = ismissing(regularisation) || (only_regulise_if_not_PSD && (minimum(eigen(mat)[1]) < 0))
    covmat = dont_regulise ? mat : regularisation(covmat, ts)

    return covmat
end
