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



@everywhere dims = 4
@everywhere syncronous = false
@everywhere with_noise = true
@everywhere grd = Int.((dims/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0)))) .* dims
fname = joinpath(fpath,  string(dims,"-",syncronous,"-",with_noise,".csv"))
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dims, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end

@everywhere dims = 16
@everywhere syncronous = false
@everywhere with_noise = true
@everywhere grd = Int.((dims/4) .* Int.(floor.(10 .^ (3.0:(1/4):5.0)))) .* dims
fname = joinpath(fpath,  string(dims,"-",syncronous,"-",with_noise,".csv"))
if isfile(fname) == false
    println("Currently doing the ", fname, " Monte Carlo." )
    ddd = @distributed (vcat) for path = pathnum
        get_convergence_errors.(Ref(functions), Ref(grd), dims, syncronous, with_noise, path)
    end
    CSV.write(fname, ddd)
end



yvar = :MAE_mean_ex_nans_mean_ex_nans
bb = combine(groupby(aa, [:ticks_per_asset, :method, :with_noise, :estimation, :variable]), :MAE_mean_ex_nans => mean_ex_nans)
bb = bb[(bb.with_noise .!= "AR1") ,: ]
plt = plot(bb, ygroup=:variable, xgroup=:estimation, Geom.subplot_grid(layer(x=:ticks_per_asset,y = yvar, color=:method, Geom.point),
           layer(x=:ticks_per_asset,y = yvar, color=:method, Geom.line), free_y_axis =true),
           Scale.x_log10, Scale.y_log10, Guide.xlabel("Average number of updates per asset"), Guide.ylabel("Mean Absolute Error"), style(key_position = :right),
           Guide.ColorKey(""))
img = PDF(joinpath(plot_folder, "both_imprecision23.pdf"), 30cm, 15cm)
draw(img, plt)
