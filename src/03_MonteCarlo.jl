"""
Generate a random path of price updates with a specified number of dimensions and ticks. There are options for whether the data is syncronous or asyncronous, the volatility of the price
processes, the refresh rate on the (exponential) arrival times of price updates, the minimum and the maximum microstructure noises.
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
