#using Revise
using HighFrequencyCovariance
using Random
using LinearAlgebra
using DataFrames
using Statistics
using Dates
using CSV

function time_and_space(dims, ticks_per_dim, numpaths = 100)
    ticks = ticks_per_dim * dims
    twister = MersenneTwister(3)
    ts, true_covar , true_micro_noise , true_update_rates = generate_random_path(dims, ticks; twister = twister)
    assets = get_assets(ts)

    dd = DataFrame()
    for m in [simple_covariance, bnhls_covariance, spectral_covariance, preaveraged_covariance, two_scales_covariance]
        println("Doing method ", m)
        m(ts,assets)
        for p in 1:numpaths
            asdf = @timed m(ts,assets)
            dd = vcat(dd, DataFrame(Method = string(m), Path = p, Time = asdf[:time], Bytes = asdf[:bytes]))
        end
    end
    dd2 = combine(groupby(dd, [:Method]), :Time => mean, :Bytes => mean)
    dd2[!, :Dims] .= dims
    dd2[!, :ticks_per_dim] .= ticks_per_dim
    return dd2
end

#Dates.format(Sys.time(), "HH:MM:SS")


dd = DataFrame()
for dims in reverse([2,4,8,16,32,64,128])
    println("dim is ", dims)
    for ticks_per_dim in [2500]
        println("ticks per dim is ", ticks_per_dim, " at time ", now())
        dd = vcat(dd,  time_and_space(dims, ticks_per_dim)  )
    end
end
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/time_and_space_complexity_up_dimensions.csv")
CSV.write(fname, dd)

dd = DataFrame()
for dims in reverse([10])
    println("dim is ", dims)
    for ticks_per_dim in [250,500,1000,2000,4000,8000,16000,32000,64000]
        println("ticks per dim is ", ticks_per_dim, " at time ", now())
        dd = vcat(dd,  time_and_space(dims, ticks_per_dim)  )
    end
end
fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/time_and_space_complexity_up_obs.csv")
CSV.write(fname, dd)
