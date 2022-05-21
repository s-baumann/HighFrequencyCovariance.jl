"""
    SortedDataFrame(df::DataFrame, time::Symbol = :Time, grouping::Symbol = :Name, value::Symbol = :Value, time_period_per_unit::Dates.Period)

This struct wraps a `DataFrame`. In the constructor function for the dataframe
we presort the data and create a mapping dict so that it is fast to subset the
DataFrame by the group.

For the constructor pass in the dataframe, name of time column, name of grouping
 column and name of value column to the constructor.
### Inputs
* `df` - The tick data
* `time` - The column of the data representing time.
* `grouping` - The column of the data representing the asset name
* `value` - The column of the data representing price/logprice/etc.
* `time_period_per_unit` - The period that one unit (in the time column) corresponds to.

### Returns
* A `SortedDataFrame`.
"""
struct SortedDataFrame{I<:Integer}
    df::DataFrame
    time::Symbol
    grouping::Symbol
    value::Symbol
    groupingrows::Dict{Symbol,Vector{I}}
    time_period_per_unit::Dates.Period
    function SortedDataFrame(df::DataFrame, time::Symbol, grouping::Symbol,
                             value::Symbol, time_period_per_unit::Dates.Period)
       df2 = sort(df, time)
       dic = Dict{Symbol,Vector{Int64}}()
       for g in unique(df2[:,grouping])
          dic[g] = findall(df2[:,grouping] .== g)
       end
       if typeof(time_period_per_unit) in [Year, Quarter, Month]
           error("time_period_per_unit cannot be a Year, Quarter or Month as it is undefined how many seconds/days/etc exist per Year, Quarter or Month.")
       end
       return new{Int64}(df2, time, grouping, value, dic, time_period_per_unit)
    end
    function SortedDataFrame(df::DataFrame, timevar::Symbol, groupingvar::Symbol, valuevar::Symbol,
                             dic::Dict{Symbol,Vector{I}}, vol_unit::Dates.Period) where I<:Integer
        return new{I}(df, timevar, groupingvar, valuevar, dic, vol_unit)
    end
    function SortedDataFrame(ts::SortedDataFrame{I}, timevar::Symbol, groupingvar::Symbol, valuevar::Symbol,
                             vol_unit::Dates.Period) where I<:Integer
        dd = ts.df
        dd[!, timevar] = time_period_ratio(vol_unit, ts.time_period_per_unit) .* dd[:, ts.time]
        rename!(dd, Dict([ts.grouping, ts.value] .=> [groupingvar, valuevar]))
        return new{I}(dd[:,[timevar, groupingvar, valuevar]], timevar, groupingvar, valuevar, ts.groupingrows, vol_unit)
    end
end


"""
    time_period_ratio(neww::Dates.Period, oldd::Dates.Period)

This calculates the ratio of the interval length between two periods. So if neww is
twice as long a period as oldd it will return a 2.0.

### Inputs
* `neww` - A time period
* `oldd` - A time period

### Returns
* A real number.
"""
function time_period_ratio(neww::Dates.Period, oldd::Dates.Period)
    return Nanosecond(neww) / Nanosecond(oldd)
end

"""
    safe_multiply_period(scalar::Real, period::Dates.Period)

This multiplies a time period by a scalar. So if period is Dates.Hour(1) and we multiply
by 2 we will get two hours (although that will be expressed in Nanosecond units).

### Inputs
* `scalar` - A real number
* `period` - A time period

### Returns
* A time period expressed in Nanosecond units.
"""
function safe_multiply_period(scalar::Real, period::Dates.Period)
    vall = Nanosecond(period).value * scalar
    return Nanosecond(floor(vall))
end

"""
    DataFrames.combine(dfs::Vector{SortedDataFrame}; timevar::Symbol = dfs[1].time, groupingvar::Symbol = dfs[1].grouping,
                 valuevar::Symbol = dfs[1].value, period::Dates.Period = dfs[1].time_period_per_unit)
Show a SortedDataFrame with a set number of rows.
### Inputs
* `dfs` - A vector of SortedDataFrames
* `timevar` - The desired name of the column representing time.
* `groupingvar` - The desired name of the column representing the asset name
* `valuevar` - The desired name of the column representing price/logprice/etc.
* `period` - The desired period that one unit (in the time column) corresponds to.
"""
function DataFrames.combine(dfs::Vector{SortedDataFrame{<:Integer}}; timevar::Symbol = dfs[1].time, groupingvar::Symbol = dfs[1].grouping,
                 valuevar::Symbol = dfs[1].value, period::Dates.Period = dfs[1].time_period_per_unit)
    dfs_newversion = SortedDataFrame.(dfs, timevar, groupingvar, valuevar, period)
    dd = vcat(map(x -> x.df, dfs_newversion)...)
    return SortedDataFrame(dd, timevar, groupingvar, valuevar, period)
end

"""
    Base.show(sdf::SortedDataFrame, number_of_rows = 10)
Show a SortedDataFrame with a set number of rows.
### Inputs
* `sdf` - The `SortedDataFrame` to show.
* `number_of_rows` - The number of rows to show.
"""
function Base.show(sdf::SortedDataFrame, number_of_rows = 10)
    println()
    println("SortedDataFrame with " , nrow(sdf.df), " rows."  )
    show(sdf.df[1:number_of_rows,[sdf.time, sdf.grouping, sdf.value]])
    println()
end


using Gadfly
"""
    Gadfly.plot(ts::SortedDataFrame)
This makes a basic plot of the assets in a `SortedDataFrame`.
### Inputs
* `ts` - The `SortedDataFrame` to plot.
"""
function Gadfly.plot(ts::SortedDataFrame)
    plt = Gadfly.plot(ts.df, x=ts.time, y=ts.value, Geom.line, color=ts.grouping)
    return plt
end



const MEANINGFUL_PRICE_DIFFERENCE = 1000*eps()

"""
    get_assets(ts::SortedDataFrame, obs_to_include::Integer = 10)

This returns a vector of all of the assets in the `SortedDataFrame` with at least
some number of observations (10 by default).
### Inputs
* `ts` - The tick data.
* `obs_to_include` - An integer for the minimum number of ticks in `ts` needed for the function to include that asset.
### Returns
* A `Vector{Symbol}` with each asset.
"""
function get_assets(ts::SortedDataFrame, obs_to_include::Integer = 10)
    all_assets = unique(ts.df[:,ts.grouping])
    assets = Array{Symbol,1}(undef,0)
    for a in all_assets
        cond1 = length(ts.groupingrows[a]) >= obs_to_include
        if sum(isnan, ts.df[ts.groupingrows[a], ts.value]) > 0
            @warn string("There are nan values for ", a)
        end
        vals = ts.df[ts.groupingrows[a], ts.value]
        minn, maxx = extrema(vals)
        # If there is no variation in price then we cannot estimate the covariance.
        # So we will drop cases with less than 1000 epsilons worth of price difference.
        # Users can avoid this behaviour by passing in assets directly rather than
        # using this get_assets function.
        cond2 = (maxx - minn) > MEANINGFUL_PRICE_DIFFERENCE
        if (cond1 && cond2) push!(assets, a) end
    end
    return sort!(assets)
end

"""
    subset_to_tick(ts::SortedDataFrame, n::Integer)

This subsets a `SortedDataFrame` to only the first n ticks.
### Inputs
* `ts` - Tick data.
* `n` - How many ticks to subset to.
### Returns
* A (smaller) `SortedDataFrame`.
"""
function subset_to_tick(ts::SortedDataFrame, n::Integer)
    newdf = ts.df[1:n,:]
    assets = get_assets(ts, 0)
    newgroupingrows = Dict{Symbol,Vector{eltype(ts.groupingrows[assets[1]])}}()
    for a in assets
        newvec = ts.groupingrows[a][ts.groupingrows[a] .<= n]
        if length(newvec) > 0 newgroupingrows[a] = newvec end
    end
    return SortedDataFrame(newdf, ts.time, ts.grouping, ts.value, newgroupingrows, ts.time_period_per_unit)
end

"""
    subset_to_time(ts::SortedDataFrame, totime::Real)

This subsets a `SortedDataFrame` to only the first observations up until some time.
### Inputs
* `ts` - Tick data.
* `totime` - Up to what time.
### Returns
* A (smaller) `SortedDataFrame`.
"""
function subset_to_time(ts::SortedDataFrame, totime::Real)
    ind = searchsortedfirst(ts.df[:,ts.time], totime) - 1
    if ind >= nrow(ts.df) return ts end
    return subset_to_tick(ts, ind)
end

"""
    duration(ts::SortedDataFrame)

The time elapsed between the first and the last tick in a `SortedDataFrame`.
### Inputs
* `ts` - Tick data.
* `in_dates_period` - In Dates.Period format or just a number for the numeric difference between first and last tick.
### Returns
* A scalar representing this duration.
"""
function duration(ts::SortedDataFrame; in_dates_period::Bool = true)
    duration_in_time_col_units = ts.df[nrow(ts.df),ts.time] - ts.df[1,ts.time]
    if !in_dates_period return duration_in_time_col_units end
    duration_in_nanoseconds = round(Nanosecond(ts.time_period_per_unit).value * duration_in_time_col_units)
    return Nanosecond(duration_in_nanoseconds)
end
