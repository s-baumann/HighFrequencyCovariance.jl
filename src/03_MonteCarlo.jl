"""
Generate a random path of price updates with a specified number of dimensions and ticks. There are options for whether the data is syncronous or asyncronous, the volatility of the price
processes, the refresh rate on the (exponential) arrival times of price updates, the minimum and the maximum microstructure noises.

### Takes
* dimensions::Integer - The number of assets
* ticks::Integer - The number of ticks to produce
* syncronous::Bool - Should ticks be syncronous (for each asset) or asyncronous
* twister::MersenneTwister - The MersenneTwister used for RNG.
* minvol::Real - The minimum volatility in sampling (only used if vols is missing)
* maxvol::Real - The maximum volatility in sampling (only used if vols is missing)
* min_refresh_rate::Real - The minimum refresh rate in sampling
* max_refresh_rate::Real - The maximum refresh rate in sampling
* min_noise_var::Real  - The minimum assetwise microstructure noise variance
* max_noise_var::Real  - The minimum assetwise microstructure noise variance
* assets::Union{Vector,Missing} = missing
* brownian_corr_matrix::Union{Hermitian,Missing} - The correlation matrix to use. This is sampled from the Inverse Wishart distribution if none is input.
* vols::Union{Vector,Missing} - The correlation matrix to use. This is sampled from the Inverse Wishart distribution if none is input.
### Returns
* A `SortedDataFrame` of tick data
* A CovarianceMatrix representing the true data generation process used in making the tick data
* A dict of microstructure_noises for each asset.
* A dict of update_rates for each asset.
"""
function generate_random_path(dimensions::Integer, ticks::Integer; syncronous::Bool = false, twister::MersenneTwister = MersenneTwister(1), minvol::Real = 0.0, maxvol::Real = 0.02,
                              min_refresh_rate::Real = 1.0, max_refresh_rate::Real = 5.0, min_noise_var::Real = 0.0, max_noise_var::Real = 0.01, assets::Union{Vector,Missing} = missing,
                              brownian_corr_matrix::Union{Hermitian,Missing} = missing, vols::Union{Vector,Missing} = missing)

    if ismissing(brownian_corr_matrix)
        wish = InverseWishart(dimensions, Matrix(Float64.(I(dimensions))))
        brownian_corr_matrix, _ = cov2cor(Hermitian(rand(twister, wish)))
    end

    vols = ismissing(vols) ? minvol .+ (maxvol .- minvol) .* rand(twister, dimensions) : vols
    assets = ismissing(assets) ? Symbol.(:asset_, 1:dimensions) : assets

    ito_integrals = Dict(assets .=> map(i -> ItoIntegral(assets[i], PE_Function(vols[i], 0.0, 0.0, 0)  ), 1:dimensions))
    ito_set_ = ItoSet(brownian_corr_matrix, assets, ito_integrals)

    covar = ForwardCovariance(ito_set_, 0.0, 1.0)
    stock_processes = Dict(assets .=> map(a -> ItoProcess(0.0, 0.0, PE_Function(0.00, 0.0, 0.0, 0), ito_integrals[a]), assets))
    update_rates = Dict(assets .=> Exponential.(    map(x -> min_refresh_rate + (max_refresh_rate - min_refresh_rate) .* x , rand(twister, dimensions))))
    microstructure_noise = Dict(assets .=> min_noise_var .+ (max_noise_var .- min_noise_var) .* rand(twister, dimensions))
    ts = syncronous ? make_ito_process_syncronous_time_series(stock_processes, covar, mean(a-> update_rates[a].θ, assets),Int(ceil(ticks/dimensions)); ito_twister = twister) : make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, ticks; timing_twister = twister, ito_twister = twister)
    ts = SortedDataFrame(ts)
    standard_normal_draws = rand(twister, Normal(), nrow(ts.df))

    normal_draws = standard_normal_draws .* map(a -> sqrt(microstructure_noise[a]), ts.df[:,ts.grouping])
    ts.df[:,ts.value] += normal_draws
    return ts, CovarianceMatrix(brownian_corr_matrix, vols, assets), microstructure_noise, update_rates
end

import StochasticIntegrals.ItoSet
"""
Convert a CovarianceMatrix into an ItoSet from the StochasticIntegrals package.
This package can then be used to do things like generate draws from the Multivariate
Gaussian corresponding to the covariance matrix and other things.

### Takes
* covariance_matrix::CovarianceMatrix{<:Real}
### Returns
* A `StochasticIntegrals.ItoSet` struct.
"""
function ItoSet(covariance_matrix::CovarianceMatrix{<:Real})
    itos = Dict{Symbol,ItoIntegral}()
    for i in 1:length(covariance_matrix.labels)
        lab = covariance_matrix.labels[i]
        voll = covariance_matrix.volatility[i]
        itos[lab] =  ItoIntegral(lab, voll)
    end
    ito_set_ = ItoSet(covariance_matrix.correlation , covariance_matrix.labels, itos)
    return ito_set_
end
