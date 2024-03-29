
"""
    simple_volatility_given_returns(returns::Vector{R}) where R<:Real
Calculates volatility using a vector of returns using the square root of squared returns.
### Inputs
* `returns` - A vector of returns
### Returns
* A `Real`
"""
function simple_volatility_given_returns(returns::Vector{R}) where R<:Real
    return sqrt(mean(returns .^ 2))
end

"""
    simple_volatility_with_grid(ts::SortedDataFrame, assets::Vector{Symbol}, time_grid)

Calculates volatility with the simple method with a specified time grid.
### Inputs
* `ts` - A `SortedDataFrame` containing the tick data
* `assets` - A `Vector` of assets that you want to calculate volatilities for.
* `time_grid` - A `Vector` of times at which to sample prices. These prices will be used in the volatility calculation.
### Returns
* A `Dict` with volatilities for each asset.
"""
function simple_volatility_with_grid(ts::SortedDataFrame, assets::Vector{Symbol}, time_grid)
    voldict =
        Dict{Symbol,eltype(ts.df[:, ts.value])}(assets .=> repeat([0.0], length(assets)))
    for a in assets
        at_times = time_grid[a]
        dd_compiled = latest_value(ts, at_times; assets = Array{Symbol}([a]))
        returns = get_returns(dd_compiled)
        vol = simple_volatility_given_returns(Array(returns[:, a]))
        minn, maxx = extrema(at_times)
        av_tick_duration = (maxx - minn) / (length(at_times) - 1)
        voldict[a] = vol / sqrt(av_tick_duration)
    end
    return voldict
end

"""
    simple_volatility(
       ts::SortedDataFrame,
       assets::Vector{Symbol} = get_assets(ts);
       time_grid::Union{Missing,Dict} = missing,
       fixed_spacing::Union{Missing,Dict,<:Real} = missing,
       use_all_obs::Bool = false,
       rough_guess_number_of_intervals::Integer = 5,
    )

Calculates volatility with the simple method.
### Inputs
* `ts` - The tick data.
* `assets` - The assets you want to estimate volatilities for.
* `time_grid` - The grid with which to calculate returns. If missing one is generated with a fixed spacing (if that is provided) or a default spacing.
* `fixed_spacing` - A spacing used to calculate a time grid. Not used if a `time_grid` is input or if `use_all_obs = true`.
* `use_all_obs` - Use all observations to estimate volatilities. Not used if a `time_grid` is provided.
* `rough_guess_number_of_intervals` - A rough number of intervals to calculate a default spacing. Not used if a `time_grid` or `fixed_spacing` is provided or if `use_all_obs = true`.
### Returns
* A `Dict` with an estimated volatility for each asset.
"""
function simple_volatility(
    ts::SortedDataFrame,
    assets::Vector{Symbol} = get_assets(ts);
    time_grid::Union{Missing,Dict} = missing,
    fixed_spacing::Union{Missing,Dict,<:Real} = missing,
    use_all_obs::Bool = false,
    rough_guess_number_of_intervals::Integer = 5,
)
    if ismissing(time_grid)
        time_grid = Dict{Symbol,Vector{eltype(ts.df[:, ts.time])}}()
        if use_all_obs
            for a in assets
                time_grid[a] = ts.df[ts.groupingrows[a], ts.time]
            end
        elseif !ismissing(fixed_spacing)
            minn, maxx = extrema(ts.df[:, ts.time])
            if isa(fixed_spacing, Dict)
                for a in assets
                    time_grid[a] = collect(minn:fixed_spacing[a]:maxx)
                end
            else
                for a in assets
                    time_grid[a] = collect(minn:fixed_spacing:maxx)
                end
            end
        else
            minn, maxx = extrema(ts.df[:, ts.time])
            n_grid = default_spacing(
                ts;
                rough_guess_number_of_intervals = rough_guess_number_of_intervals,
            )
            for a in assets
                time_grid[a] = collect(minn:n_grid[a]:maxx)
            end
        end
    end
    return simple_volatility_with_grid(ts, assets, time_grid)
end

"""
    default_spacing(
        ts::SortedDataFrame;
        rough_guess_number_of_intervals::Integer = 5,
        T::Real = duration(ts; in_dates_period = false),
    )

Calculates a default spacing between returns to use.
This comes from the equation at section 1.2.3 of Zhang, Mykland, Ait-Sahalia 2005.
### Inputs
* `ts` - The tick data.
* `rough_guess_number_of_intervals` - A rough estimate of how many intervals to split the tick data into. This is used in a first pass to estimate the optimal interval spacing.
* `T` - The duration of the tick data.
### Returns
* A scalar representing the optimal interval spacing.
"""
function default_spacing(
    ts::SortedDataFrame;
    rough_guess_number_of_intervals::Integer = 5,
    T::Real = duration(ts; in_dates_period = false),
)
    rough_vol_guess, rough_micro_guess =
        two_scales_volatility(ts; num_grids = rough_guess_number_of_intervals)
    n_guess = Dict{Symbol,eltype(ts.df[:, ts.value])}()
    for a in keys(rough_vol_guess)
        # Note that if opur guess is that there is no microstructure noise then the formula from this
        # paper gives an infinite number of intervals. So in this case we will just choose 20 which is
        # an arbitrary choice but seems sensible. Advanced users can choose their own spacing to avoid this.
        if (rough_micro_guess[a] < 0.0000001)
            n_guess[a] = 20
        else
            n_guess[a] =
                ((T / (4 * rough_micro_guess[a]^2)) * T * rough_vol_guess[a]^4)^(1 / 3)
        end
    end
    return n_guess
end
