using DataFrames
using LinearAlgebra
using Statistics: std, var, mean, cov
using HighFrequencyCovariance
using CSV
using Plots

using UnivariateFunctions
using StochasticIntegrals
using Distributions
using Random
using Distributed

fpath = @__DIR__
fpath = joinpath(fpath, "data")


num_cores = 8
addprocs(num_cores)
@everywhere using HighFrequencyCovariance, Random, DataFrames, Dates, Distributions
@everywhere functions = [:bnhls_covariance, :two_scales_covariance, :spectral_covariance,  :simple_covariance, :preaveraged_covariance ]
@everywhere pathnum = 1:200
@everywhere grd = Int.(floor.(10 .^ (3.0:(1/4):6.0)))
@everywhere time_period_per_unit = Dates.Hour(1)

@everywhere function get_convergence_errors(functions, grd, dimensions, syncronous, with_noise, pathnum)
    println("Doing pathnum = ", pathnum, " at time ", Dates.now())
    ticks = maximum(grd)
    twister = MersenneTwister(pathnum)
    ts, true_covar, micro_noise, update_rates = generate_random_path(dimensions, ticks; time_period_per_unit = time_period_per_unit, syncronous = syncronous, twister = twister, max_noise_var = with_noise * 0.01)
    dd = DataFrame()
    detail_vars = Dict{Symbol,Any}([:dimensions, :syncronous, :with_noise, :pathnum] .=> Any[dimensions, syncronous, with_noise, pathnum])
    for gridlen in grd
        sub_ts = subset_to_tick(ts, gridlen)
        for method in functions
            covar = estimate_covariance(sub_ts, get_assets(sub_ts), method )
            detail_vars2 = deepcopy(detail_vars)
            detail_vars2[:method] = string(method)
            detail_vars2[:number_of_paths] = gridlen
            newdf = HighFrequencyCovariance.to_dataframe(covar, detail_vars2)
            dd = append!(dd, newdf)
        end
    end
    newdf = HighFrequencyCovariance.to_dataframe(true_covar, detail_vars; delete_duplicate_correlations = false)
    rename!(newdf, :value => :true_value)
    select!(newdf, Not([:vol_period_units, :vol_period]))
    merged = leftjoin(dd, newdf, on = [ :asset1, :asset2, :variable, :syncronous, :pathnum, :dimensions, :with_noise])

    number_we_should_have = length(functions) * ((dimensions * (dimensions-1))/2 + dimensions) * length(grd)
    if nrow(merged) != number_we_should_have
        error(string("We did not get all the rows we should have. We should have ", number_we_should_have, " while we only have ", nrow(merged) ))
    end
    return merged
end



@everywhere dimensions = 4
@everywhere syncronous = false
@everywhere with_noise = true
@everywhere grd = Int.( (dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))  ) ) .* dimensions
fname = joinpath(fpath,  string(dimensions,"-",syncronous,"-",with_noise,".csv"))
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end


@everywhere dimensions = 20
@everywhere syncronous = false
@everywhere with_noise = true
@everywhere grd = Int.( (dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))  ) ) .* dimensions
fname = joinpath(fpath,  string(dimensions,"-",syncronous,"-",with_noise,".csv"))
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end
