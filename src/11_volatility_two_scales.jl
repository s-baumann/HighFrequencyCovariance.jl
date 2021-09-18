
function vol_given_values_and_times(vals::Vector, times::Vector, asset::Symbol)
    duration = maximum(times) - minimum(times)
    returns = simple_differencing(vals[2:end], vals[1:(end-1)])
    return sqrt(sum(returns .^ 2)/duration)
end

"""
    default_num_grids(ts::SortedDataFrame)

This gives a default number of intervals to divide a series of ticks over for the purposes of estimating returns and volatility.
### Inputs
* `ts` - The tick data.
### Returns
* An integer for the number of intervals.
"""
function default_num_grids(ts::SortedDataFrame)
    min_ticks = minimum(map( a -> length(ts.groupingrows[a]) , collect(keys(ts.groupingrows)) ))
    return Int(max(floor(min_ticks / 100), 3))
end

"""
    two_scales_volatility(vals::Vector, times::Vector, asset::Symbol, num_grids::Real)

Calculates volatility with the two scales method of Zhang, Mykland, Ait-Sahalia 2005. The amount of time for the grid spacing is by default this is a tenth of the total duration
by default. If this doesn't make sense for your use of it then choose a spacing at which you expect the effect of microstructure noise will be small.
### Inputs
* `vals` - The prices at each instant in time.
* `times` - The times corresponding to each element in `vals`.
* `asset` - The name of the asset.
* `num_grids` - Number of grids used in order in two scales estimation.
### Returns
* A scalar for the estimated volatility of the asset.
* A scalar for the estimated microstructure noise variance.


    two_scales_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
                          num_grids::Real = default_num_grids(ts))

Calculates volatility with the two scales method of Zhang, Mykland, Ait-Sahalia 2005. The amount of time for the grid spacing is by default this is a tenth of the total duration
by default. If this doesn't make sense for your use of it then choose a spacing at which you expect the effect of microstructure noise will be small.
### Inputs
* `ts` - The tick data.
* `assets` - The assets you want to estimate volatilities for.
* `num_grids` - Number of grids used in order in two scales estimation.
### Returns
* A  `Dict` with estimated volatilities for each asset.
* A  `Dict` with estimated microstructure noise variances for each asset.

### References
Zhang L, Mykland PA, Aït-Sahalia Y (2005). "A Tale of Two Time Scales: Determining Integrated Volatility with Noisy High-Frequency Data." Journal of the American Statistical Association, 100(472), 1394–1411. ISSN 01621459. doi:10.1198/016214505000000169.
"""
function two_scales_volatility(vals::Vector, times::Vector, asset::Symbol, num_grids::Real)
    dura  = maximum(times) - minimum(times)
    if (dura < eps()) | (length(vals) < 10) return NaN, NaN end
    num_grids = Int(  min( floor(length(times)/4),   max( 2,  floor(num_grids)  ) ))
    avg_vol   = mean(map(i ->  vol_given_values_and_times(vals[i:num_grids:end], times[i:num_grids:end], asset), 1:num_grids ))
    all_vol   = vol_given_values_and_times(vals, times, asset)

    pure_vol  = ((1- 1/num_grids)^(-1)) * ( avg_vol - (1/num_grids) *all_vol   )
    noise     = (1/(2*length(vals))) * ((all_vol^2)*dura - (pure_vol^2)*dura )
    return pure_vol, noise
end

function two_scales_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
                               num_grids::Real = default_num_grids(ts))
    vols = Dict{Symbol,eltype(ts.df[:,ts.value])}()
    micro_noise_var = Dict{Symbol,eltype(ts.df[:,ts.value])}()
    for a in assets
        vals = ts.df[ts.groupingrows[a],ts.value]
        times = ts.df[ts.groupingrows[a],ts.time]
        if length(vals) < 10
            @warn "There was not enough data to use the two_scales_volatility method. Returning NaN."
            vols[a] = NaN
            micro_noise_var[a] = NaN
        else
            pure_vol, noise = two_scales_volatility(vals, times, a, num_grids)
            vols[a] = pure_vol
            micro_noise_var[a] = max(0.0,noise)
        end
    end
    return vols, micro_noise_var
end
