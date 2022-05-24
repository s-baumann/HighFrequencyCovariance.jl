
"""
    latest_value(
        ts::SortedDataFrame,
        at_times::Vector{<:Real};
        assets::Vector{Symbol} = get_assets(ts),
    )

Get the latest price at a each input time.
### Inputs
* `ts` - The tick data.
* `at_times` - The times you want the latest prices for.
* `assets` - The assets you want latest prices for.
### Returns
* A `DataFrame`. Rows are for each time specified in at_times. Columns are for each asset.
"""
function latest_value(
    ts::SortedDataFrame,
    at_times::Vector{<:Real};
    assets::Vector{Symbol} = get_assets(ts),
)
    dd = DataFrame(Time = at_times)
    for a in assets
        indx = searchsortedlast_both(ts.df[ts.groupingrows[a], ts.time], at_times)
        dd[!, a] = ts.df[ts.groupingrows[a][indx], ts.value]
    end
    return dd
end

"""
    searchsortedlast_both(reference::Vector, indices::Vector)

If you have two sorted vectors. You want to find the index of each "indices" in
the vector "reference". then this is much faster than doing something like
map( x -> searchsortedlast(reference, x), indices) as it only goes once through
the reference vector.
"""
function searchsortedlast_both(reference::Vector, indices::Vector)
    len = length(indices)
    lenj = length(reference)
    inds = Array{Int64,1}(repeat([lenj], len))
    inds[1] = max(1, searchsortedlast(reference, indices[1]))
    if len == 1
        return inds
    end
    j = inds[1]
    i = 2
    while true
        if reference[j+1] > indices[i]
            inds[i] = j
            i += 1
        else
            j += 1
        end
        if (i > len) || (j == lenj)
            return inds
        end
    end
end

"""
    random_value_in_interval(
       ts::SortedDataFrame,
       at_times::Vector{<:Real};
       assets::Vector{Symbol} = get_assets(ts),
       twister_arb_value_in_interval::MersenneTwister = MersenneTwister(2604),
    )

Get a random value in an interval. So if you input times 1,7,8 then for the second entry it will pick a random update (if any exist) between times 1 and 7.
### Inputs
* `ts` - The tick data.
* `at_times` - The times that seperate the intervals of interest.
* `assets` - The assets of interest.
* `twister_arb_value_in_interval` - The RNG used in selecting the random interval.
### Returns
* A `DataFrame` with prices for each asset from random ticks in each interval.
"""
function random_value_in_interval(
    ts::SortedDataFrame,
    at_times::Vector{<:Real};
    assets::Vector{Symbol} = get_assets(ts),
    twister_arb_value_in_interval::MersenneTwister = MersenneTwister(2604),
)
    dd = DataFrame(Time = at_times)
    for a in assets
        indx = [1, searchsortedlast_both(ts.df[ts.groupingrows[a], ts.time], at_times)...]
        rand_indx = map(
            i -> rand(
                twister_arb_value_in_interval,
                min(indx[i+1], indx[i] + 1):indx[i+1],
                1,
            )[1],
            1:(length(indx)-1),
        )
        dd[!, a] = ts.df[ts.groupingrows[a][rand_indx], ts.value]
    end
    return dd
end


"""
    next_tick(ts::SortedDataFrame, from_index::I;
                       assets::Vector{Symbol} = get_assets(ts))  where R<:Real where I<:Integer

This gets the next tick by which every asset has a refreshed price after a certain row index.
### Inputs
* `ts` - The tick data.
* `from_index` - The index in your ts.df to start looking from.
* `assets` - The vector of assets that you want to get a refresh time by which each has a refreshed price.
### Returns
* A `Real` or `Missing` for the refresh time. If it is a real it is the time. If one asset did not update in your data then a missing is returned.
* An `Integer` or `Missing` for the refresh tick. for what index in your data the refresh happened by. If one asset did not refresh this will be a missing.
"""
function next_tick(ts::SortedDataFrame, from_index::I;
                   assets::Vector{Symbol} = get_assets(ts))  where R<:Real where I<:Integer
    inds = Array{I,1}()
    for a in assets
        ind = searchsortedfirst(ts.groupingrows[a], from_index)
        if (ind > length(ts.groupingrows[a])) return missing, missing end
        push!(inds, ts.groupingrows[a][ind])
    end
    refresh_index = maximum(inds)
    refresh_time = ts.df[refresh_index, ts.time]
    return refresh_time, refresh_index
end


"""
    get_all_refresh_times(
        ts::SortedDataFrame,
        assets::Vector{Symbol} = get_assets(ts);
        start_time::R = minimum(ts.df[:, ts.time]),
    ) where R<:Real

Get a vector of all refresh times when all assets have an updated price.
So if there are assets A and B that trade at times (1,5,6,7,10) and (2,5,7,9)
then the refresh times are (2,5,7,10) as at these four times there are updated
prices for all assets that have happened since the previous refresh time.
### Inputs
* `ts` - The tick data.
* `assets` - The assets of interest.
* `start_time` - From what time should we start looking for updated prices.
### Returns
* A `Vector` of refresh times.
"""
function get_all_refresh_times(
    ts::SortedDataFrame,
    assets::Vector{Symbol} = get_assets(ts);
    start_time::R = minimum(ts.df[:, ts.time]),
) where R<:Real
    ticks = Vector{R}()
    start_ind = searchsortedfirst(ts.df[:, ts.time], start_time) - 1
    while true
        new_tick, start_ind = next_tick(ts, start_ind + 1; assets = assets)
        if ismissing(new_tick)
            return ticks
        end
        push!(ticks, new_tick)
    end
end
