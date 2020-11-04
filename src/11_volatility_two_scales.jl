

function vol_given_values_and_times(vals::Vector, times::Vector, asset::Symbol, return_calc::Function)
    duration = maximum(times) - minimum(times)
    time_diffs = Array(times[2:end] .- times[1:(end-1)])
    returns = return_calc(vals[2:end], vals[1:(end-1)], time_diffs , asset)
    return sqrt(sum(returns .^ 2)/duration)
end

function two_scales_volatility(vals::Vector, times::Vector, asset::Symbol, grid_spacing::Real, return_calc::Function = simple_differencing)
    dura  = maximum(times) - minimum(times)
    if (dura < eps()) | (length(vals) < 10)
        return NaN, NaN
    end
    num_grids = Int(max(2, floor((grid_spacing/dura) * length(vals))))
    avg_vol   = mean(map(i ->  vol_given_values_and_times(vals[i:num_grids:end], times[i:num_grids:end], asset, return_calc), 1:num_grids ))
    all_vol   = vol_given_values_and_times(vals, times, asset, return_calc)

    pure_vol  = ((1- 1/num_grids)^(-1)) * ( avg_vol - (1/num_grids) *all_vol   )
    noise     = (1/(2*length(vals))) * ((all_vol^2)*dura - (pure_vol^2)*dura )
    return pure_vol, noise
end

"""
Calculates volatility with the two scales method of Zhang, Mykland, Ait-Sahalia 2005. The amount of time for the grid spacing is by default this is a tenth of the total duration
by default. If this doesn't make sense for your use of it then choose a spacing at which you expect the effect of microstructure noise will be small.
"""
function two_scales_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); grid_spacing::Real = duration(ts)/10, return_calc::Function = simple_differencing)
    vols = Dict{Symbol,eltype(ts.df[:,ts.value])}()
    micro_noise_var = Dict{Symbol,eltype(ts.df[:,ts.value])}()
    for a in assets
        vals = ts.df[ts.groupingrows[a],ts.value]
        times = ts.df[ts.groupingrows[a],ts.time]
        if length(vals) < 10
            vols[a] = NaN
            micro_noise_var[a] = NaN
        else
            pure_vol, noise = two_scales_volatility(vals, times, a, grid_spacing, return_calc)
            vols[a] = pure_vol
            micro_noise_var[a] = max(0.0,noise)
        end
    end
    return vols, micro_noise_var
end
