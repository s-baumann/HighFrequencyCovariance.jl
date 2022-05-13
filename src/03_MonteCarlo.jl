using StochasticIntegrals

"""
    make_random_psd_matrix_from_wishart(num_assets::Integer, rng::Union{MersenneTwister,StableRNG} = MersenneTwister(1))

Make a random psd matrix from the inverse wishart distribution.

### Inputs
* `num_assets` - The number of assets.
* `rng` - The Random.MersenneTwister or StableRNGs.Stable used for RNG.
### Returns
* A `Hermitian`
"""
function make_random_psd_matrix_from_wishart(num_assets::Integer, rng::Union{MersenneTwister,StableRNG} = MersenneTwister(1))
    wish = InverseWishart(num_assets, Matrix(Float64.(I(num_assets))))
    newmat = Hermitian(rand(rng, wish))
    return newmat
end

"""
    generate_random_path(dimensions::Integer, ticks::Integer;
                         syncronous::Bool = false,
                         rng::Union{MersenneTwister,StableRNG} = MersenneTwister(1),
                         vol_dist::Distribution = Uniform(0.1/sqrt(252 * 8 * 3600), 0.5/sqrt(252 * 8 * 3600)),
                         refresh_rate_dist::Distribution    = Uniform(0.5, 5.0),
                         time_period_per_unit::Dates.Period = Second(1),
                         micro_noise_dist::Distribution     = Uniform(vol_dist.a * sqrt(time_period_ratio(Minute(5), time_period_per_unit)), vol_dist.b * sqrt(time_period_ratio(Minute(5), time_period_per_unit))),
                         assets::Union{Vector,Missing}      = missing,
                         brownian_corr_matrix::Union{Hermitian,Missing} = missing,
                         vols::Union{Vector,Missing}        = missing)

Generate a random path of price updates with a specified number of dimensions and ticks. There are options for whether the data is syncronous or asyncronous, the volatility of the price
processes, the refresh rate on the (exponential) arrival times of price updates, the minimum and the maximum microstructure noises.

Note the defaults are chosen to reflect a highcap stock with annualised volatility between 10% and 50%.
The standard deviation of microstructure noise is of the same order of magnitude as
5 minutes standard deviation of return. `vol * sqrt(60*5)` if vol is in seconds.
Refreshed ticks every 0.5-5 seconds (in expectation).

### Inputs
* `dimensions` - The number of assets.
* `ticks` - The number of ticks to produce.
* `syncronous` - Should ticks be syncronous (for each asset) or asyncronous.
* `rng` - The Random.MersenneTwister or StableRNGs.Stable used for RNG.
* `vol_dist` - The distribution to draw volatilities from (only used if vols is missing).
* `refresh_rate_dist` - The distribution to draw refresh rates (exponential distribution rates) from.
* `time_period_per_unit` - What time period should the time column correspond to.
* `micro_noise_dist`  - The distribution to draw assetwise microstructure noise standard deviations are drawn from.
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
                              rng::Union{MersenneTwister,StableRNG} = MersenneTwister(1),
                              vol_dist::Distribution = Uniform(0.1/sqrt(252 * 8 * 3600), 0.5/sqrt(252 * 8 * 3600)),
                              refresh_rate_dist::Distribution    = Uniform(0.5, 5.0),
                              time_period_per_unit::Dates.Period = Second(1),
                              micro_noise_dist::Distribution     = Uniform(vol_dist.a * sqrt(time_period_ratio(Minute(5), time_period_per_unit)), vol_dist.b * sqrt(time_period_ratio(Minute(5), time_period_per_unit))),
                              assets::Union{Vector,Missing}      = missing,
                              brownian_corr_matrix::Union{Hermitian,Missing} = missing,
                              vols::Union{Vector,Missing}        = missing,
                              rng_timing::Union{MersenneTwister,StableRNG} = MersenneTwister(1))
    if (ismissing(assets) == false) && (dimensions != length(assets))
        error("If you input asset names then the number of asset names must be of length equal to the dimensions input.")
    end

    if ismissing(brownian_corr_matrix)
        brownian_corr_matrix, _ = cov2cor(make_random_psd_matrix_from_wishart(dimensions, rng))
    end

    vols = ismissing(vols) ? rand(rng, vol_dist, dimensions) : vols
    assets = ismissing(assets) ? Symbol.(:asset_, 1:dimensions) : assets

    ito_integrals = Dict(assets .=> map(i -> ItoIntegral(assets[i], PE_Function(vols[i], 0.0, 0.0, 0)  ), 1:dimensions))
    ito_set_ = ItoSet(brownian_corr_matrix, assets, ito_integrals)

    covar = StochasticIntegrals.SimpleCovariance(ito_set_, 0.0, 1.0; calculate_inverse = false, calculate_determinant = false)
    stock_processes = Dict(assets .=> map(a -> ItoProcess(0.0, 0.0, PE_Function(0.00, 0.0, 0.0, 0), ito_integrals[a]), assets))
    update_rates = Dict(assets .=> Exponential.(rand(rng, refresh_rate_dist, dimensions)))
    microstructure_noise = Dict(assets .=> rand(rng, micro_noise_dist, dimensions) .^ 2)
    rng_obj = convert_to_stochastic_integrals_type(rng, dimensions)
    ts = DataFrame()
    if syncronous
        ts = StochasticIntegrals.make_ito_process_syncronous_time_series(stock_processes, covar, mean(a-> update_rates[a].θ, assets), Int(ceil(ticks/dimensions)); number_generator = rng_obj)
    else
        ts = StochasticIntegrals.make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, ticks; timing_twister = rng_timing, ito_number_generator = rng_obj)
    end
    ts = SortedDataFrame(ts, :Time, :Name, :Value, time_period_per_unit)

    standard_normal_draws = rand(rng, Normal(), nrow(ts.df))

    normal_draws = standard_normal_draws .* map(a -> sqrt(microstructure_noise[a]), ts.df[:,ts.grouping])
    ts.df[:,ts.value] += normal_draws
    return ts, CovarianceMatrix(brownian_corr_matrix, vols, assets, time_period_per_unit), microstructure_noise, update_rates
end

function convert_to_stochastic_integrals_type(x::MersenneTwister, num::Integer)
    return StochasticIntegrals.Mersenne(x, num)
end
function convert_to_stochastic_integrals_type(x::StableRNG, num::Integer)
    return StochasticIntegrals.Stable_RNG(x, num)
end

"""
    StochasticIntegrals.ItoSet(covariance_matrix::CovarianceMatrix{<:Real})

Convert a `CovarianceMatrix` into an `ItoSet` from the StochasticIntegrals package.
This package can then be used to do things like generate draws from the Multivariate
Gaussian corresponding to the covariance matrix and other things.
### Inputs
* `covariance_matrix` - The `CovarianceMatrix` that you want to convert into an `StochasticIntegrals.ItoSet`
### Returns
* A `StochasticIntegrals.ItoSet` struct.
### Example
    using Dates
    covar = CovarianceMatrix(make_random_psd_matrix_from_wishart(5), rand(5), [:A,:B,:C,:D,:E], Dates.Hour(1))
    iset = ItoSet(covar)
    # To see how this is used for something useful you can look at the get_draws function.
"""
function StochasticIntegrals.ItoSet(covariance_matrix::CovarianceMatrix{<:Real})
    itos = Dict{Symbol,ItoIntegral}()
    for i in 1:length(covariance_matrix.labels)
        lab = covariance_matrix.labels[i]
        voll = covariance_matrix.volatility[i]
        itos[lab] =  ItoIntegral(lab, voll)
    end
    ito_set_ = StochasticIntegrals.ItoSet(covariance_matrix.correlation , covariance_matrix.labels, itos)
    return ito_set_
end

"""
    StochasticIntegrals.get_draws(covariance_matrix::CovarianceMatrix{<:Real}, num::Integer; number_generator::NumberGenerator = Mersenne(MersenneTwister(1234), length(covar.covariance_labels_)), antithetic_variates = false)
get pseudorandom draws from a `CovarianceMatrix` struct. This is basically a convenience wrapper over StochasticIntegrals.get_draws which does the necessary constructing of the structs of that package.
If the `antithetic_variates` control is set to true then every second set of draws will be antithetic to the previous.
If you want to do something like Sobol sampling you can change the number_generator. See StochasticIntegrals to see what is available (and feel free to make new ones and put in Pull Requests)
### Inputs
* `covar` - An `CovarianceMatrix` struct that you want to draw from.
* `num`- The number of draws you want
* `number_generator`  - A `NumberGenerator` struct that can be queried for a series of unit interval vectors that are then transformed by the covariance matrix into draws.
* `antithetic_variates` - A boolean indicating if antithetic variates should be used (every second draw is made from 1 - uniformdraw of previous)
### Returns
* A `Vector` of `Dict`s of draws. Note you can convert this to a dataframe or array with `StochasticIntegrals.to_dataframe` or `StochasticIntegrals.to_array`.
"""
function StochasticIntegrals.get_draws(covariance_matrix::CovarianceMatrix{<:Real}, num::Integer; number_generator::NumberGenerator = Mersenne(MersenneTwister(1234), length(covariance_matrix.labels)), antithetic_variates = false)
    iset = ItoSet(covariance_matrix)
    # And below shows how this might be used to generate random draws.
    scovar = StochasticIntegrals.SimpleCovariance(iset, 0.0, 1.0; calculate_inverse = false, calculate_determinant = false)
    draws = StochasticIntegrals.get_draws(scovar, num)
    return draws
end
