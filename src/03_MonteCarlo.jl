import StochasticIntegrals.ItoSet

"""
    generate_random_path(dimensions::Integer, ticks::Integer; syncronous::Bool = false,
                         twister::MersenneTwister = MersenneTwister(1),
                         vol_dist::Distribution = Uniform(0.1/sqrt(252 * 8 * 3600), 0.5/sqrt(252 * 8 * 3600)),
                         refresh_rate_dist::Distribution    = Uniform(0.5, 5.0),
                         time_period_per_unit::Dates.Period = Second(1),
                         micro_noise_dist::Distribution     = Uniform(vol_dist.a * sqrt(time_period_ratio(Minute(15), time_period_per_unit)), vol_dist.b * sqrt(time_period_ratio(Minute(15), time_period_per_unit))),
                         assets::Union{Vector,Missing}      = missing,
                         brownian_corr_matrix::Union{Hermitian,Missing} = missing,
                         vols::Union{Vector,Missing}        = missing)

Generate a random path of price updates with a specified number of dimensions and ticks. There are options for whether the data is syncronous or asyncronous, the volatility of the price
processes, the refresh rate on the (exponential) arrival times of price updates, the minimum and the maximum microstructure noises.

Note the defaults are chosen to reflect a highcap stock with annulised volatility between 10% and 50%.
Microstructure noise is equal to 15 minutes standard deviation of return. vol * sqrt(60*15)
Refreshed ticks every 0.5-5 seconds (in expectation).

### Inputs
* `dimensions` - The number of assets.
* `ticks` - The number of ticks to produce.
* `syncronous` - Should ticks be syncronous (for each asset) or asyncronous.
* `twister` - The MersenneTwister used for RNG.
* `vol_dist` - The distribution to draw volatilities from (only used if vols is missing).
* `refresh_rate_dist` - The distribution to draw refresh rates (exponential distribution rates) from.
* `time_period_per_unit` - What time period should the time column correspond to.
* `micro_noise_dist`  - The distribution to draw assetwise microstructure noise variances from.
* `assets` - The names of the assets that you want to use. The length of this must be equal to the `dimensions` input.
* `brownian_corr_matrix` - The correlation matrix to use. This is sampled from the Inverse Wishart distribution if none is input.
* `vols` - The volatilities to use. These are sampled  from the uniform distribution between `min_noise_var` and `max_noise_var`.
### Returns
* A `SortedDataFrame` of tick data.
* A `CovarianceMatrix` representing the true data generation process used in making the tick data.
* A `Dict` of microstructure noise variances for each asset.
* A `Dict` of update rates for each asset.
"""
function generate_random_path(dimensions::Integer, ticks::Integer;
                              syncronous::Bool = false,
                              twister::MersenneTwister = MersenneTwister(1),
                              vol_dist::Distribution = Uniform(0.1/sqrt(252 * 8 * 3600), 0.5/sqrt(252 * 8 * 3600)),
                              refresh_rate_dist::Distribution    = Uniform(0.5, 5.0),
                              time_period_per_unit::Dates.Period = Second(1),
                              micro_noise_dist::Distribution     = Uniform(vol_dist.a * sqrt(time_period_ratio(Minute(15), time_period_per_unit)), vol_dist.b * sqrt(time_period_ratio(Minute(15), time_period_per_unit))),
                              assets::Union{Vector,Missing}      = missing,
                              brownian_corr_matrix::Union{Hermitian,Missing} = missing,
                              vols::Union{Vector,Missing}        = missing)
    if (ismissing(assets) == false) && (dimensions != length(assets))
        error("If you input asset names then the number of asset names must be of length equal to the dimensions input.")
    end

    if ismissing(brownian_corr_matrix)
        wish = InverseWishart(dimensions, Matrix(Float64.(I(dimensions))))
        brownian_corr_matrix, _ = cov2cor(Hermitian(rand(twister, wish)))
    end

    vols = ismissing(vols) ? rand(twister, vols, dimensions) : vols
    assets = ismissing(assets) ? Symbol.(:asset_, 1:dimensions) : assets

    ito_integrals = Dict(assets .=> map(i -> ItoIntegral(assets[i], PE_Function(vols[i], 0.0, 0.0, 0)  ), 1:dimensions))
    ito_set_ = ItoSet(brownian_corr_matrix, assets, ito_integrals)

    covar = StochasticIntegrals.SimpleCovariance(ito_set_, 0.0, 1.0; calculate_inverse = false, calculate_determinant = false)
    stock_processes = Dict(assets .=> map(a -> ItoProcess(0.0, 0.0, PE_Function(0.00, 0.0, 0.0, 0), ito_integrals[a]), assets))
    update_rates = Dict(assets .=> Exponential.(rand(twister, refresh_rate_dist, dimensions)))
    microstructure_noise = Dict(assets .=> rand(twister, micro_noise_dist, dimensions))
    ts = syncronous ? StochasticIntegrals.make_ito_process_syncronous_time_series(stock_processes, covar, mean(a-> update_rates[a].θ, assets), Int(ceil(ticks/dimensions)); ito_twister = twister) : StochasticIntegrals.make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, ticks; timing_twister = twister, ito_twister = twister)
    ts = SortedDataFrame(ts, :Time, :Name, :Value, time_period_per_unit)

    standard_normal_draws = rand(twister, Normal(), nrow(ts.df))

    normal_draws = standard_normal_draws .* map(a -> sqrt(microstructure_noise[a]), ts.df[:,ts.grouping])
    ts.df[:,ts.value] += normal_draws
    return ts, CovarianceMatrix(brownian_corr_matrix, vols, assets, time_period_per_unit), microstructure_noise, update_rates
end

"""
    ItoSet(covariance_matrix::CovarianceMatrix{<:Real})

Convert a `CovarianceMatrix` into an `ItoSet` from the StochasticIntegrals package.
This package can then be used to do things like generate draws from the Multivariate
Gaussian corresponding to the covariance matrix and other things.
### Inputs
* `covariance_matrix` - The `CovarianceMatrix` that you want to convert into an `StochasticIntegrals.ItoSet`
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
    ito_set_ = StochasticIntegrals.ItoSet(covariance_matrix.correlation , covariance_matrix.labels, itos)
    return ito_set_
end
