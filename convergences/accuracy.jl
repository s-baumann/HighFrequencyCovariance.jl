using Revise
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

num_cores = 8
addprocs(num_cores)
@everywhere using HighFrequencyCovariance, Random, DataFrames, Dates
@everywhere functions = [bnhls_covariance, two_scales_covariance, spectral_covariance,  simple_covariance, preaveraged_covariance ]
@everywhere pathnum = 1:400
@everywhere grd = Int.(floor.(10 .^ (3.0:(1/4):5.0)))

@everywhere function get_convergence_errors(functions, grd, dimensions, syncronous, with_noise, pathnum)
    println("Doing pathnum = ", pathnum, " at time ", Dates.now())
    ticks = maximum(grd)
    twister = MersenneTwister(pathnum)
    ts, true_covar, micro_noise, update_rates = generate_random_path(dimensions, ticks; syncronous = syncronous, twister = twister, max_noise_var = with_noise * 0.01)
    dd = DataFrame()
    detail_vars = Dict{Symbol,Any}([:dimensions, :syncronous, :with_noise, :pathnum] .=> [dimensions, syncronous, with_noise, pathnum])
    for gridlen in grd
        sub_ts = subset_to_tick(ts, gridlen)
        for method in functions
            covar = method(sub_ts)
            detail_vars2 = deepcopy(detail_vars)
            detail_vars2[:method] = string(method)
            detail_vars2[:number_of_paths] = gridlen
            newdf = HighFrequencyCovariance.to_dataframe(covar, detail_vars2)
            dd = append!(dd, newdf)
        end
    end
    newdf = HighFrequencyCovariance.to_dataframe(true_covar, detail_vars; delete_duplicate_correlations = false)
    rename!(newdf, :value => :true_value)
    merged = leftjoin(dd, newdf, on = [ :asset1, :asset2, :variable, :syncronous, :pathnum, :dimensions, :with_noise])
    return merged
end
@everywhere dimensions = 4
@everywhere syncronous = false
@everywhere with_noise = true
@everywhere grd = Int.(floor.(10 .^ (3.0:(1/4):5.0)))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end

@everywhere dimensions = 16
@everywhere syncronous = false
@everywhere with_noise = true
@everywhere grd = Int.((dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end






@everywhere dimensions = 4
@everywhere syncronous = true
@everywhere with_noise = true
@everywhere grd = Int.((dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end

@everywhere dimensions = 16
@everywhere syncronous = true
@everywhere with_noise = true
@everywhere grd = Int.((dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end





@everywhere dimensions = 4
@everywhere syncronous = false
@everywhere with_noise = false
@everywhere grd = Int.((dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end

@everywhere dimensions = 16
@everywhere syncronous = false
@everywhere with_noise = false
@everywhere grd = Int.((dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end



@everywhere dimensions = 4
@everywhere syncronous = true
@everywhere with_noise = false
@everywhere grd = Int.((dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    #ddd = reduce(vcat, conv)
    CSV.write(fname, ddd)
end

@everywhere dimensions = 16
@everywhere syncronous = true
@everywhere with_noise = false
@everywhere grd = Int.((dimensions/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0))))
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/",  dimensions,"-",syncronous,"-",with_noise,".csv")
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dimensions, syncronous, with_noise, path)
    end
    #ddd = reduce(vcat, conv)
    CSV.write(fname, ddd)
end
