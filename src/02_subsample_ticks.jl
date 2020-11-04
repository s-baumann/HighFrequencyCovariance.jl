"""
Get the latest price at a each input time.
"""
function latest_value(ts::SortedDataFrame, at_times::Vector{R}; assets::Vector{Symbol} = get_assets(ts)) where R<:Real
    dd = DataFrame(Time = at_times)
    for a in assets
        indx = searchsortedlast_both(ts.df[ts.groupingrows[a],ts.time], at_times)
        dd[!,a] = ts.df[ts.groupingrows[a][indx]  , ts.value]
    end
    return dd
end

"""
If you have two sorted vectors. You want to find the index of each "indices" in the vector "reference". then this is much faster than
doing something like map( x -> searchsortedlast(reference, x), indices) as it only goes once through the reference vector.
"""
function searchsortedlast_both(reference::Vector, indices::Vector)
    len = length(indices)
    lenj = length(reference)
    inds = Array{Int64,1}(repeat([lenj], len))
    inds[1] = max(1,searchsortedlast(reference, indices[1]))
    if len == 1 return inds end
    j = inds[1]
    i = 2
    while true
        if reference[j+1] > indices[i]
            inds[i] = j
            i += 1
        else
            j += 1
        end
        if (i > len) | (j == lenj) return inds end
    end
end









"""
Get a random value in an interval. So if you input times 1,7,8 then for the second entryy it will pick a random update (if any exist) between times 1 and 7.
"""
function random_value_in_interval(ts::SortedDataFrame, at_times::Vector{R}; assets::Vector{Symbol} = get_assets(ts), twister_arb_value_in_interval::MersenneTwister = MersenneTwister(2604)) where R<:Real
    dd = DataFrame(Time = at_times)
    for a in assets
        #dff = ts.df[ts.groupingrows[a],:]
        #NN = nrow(dff)
        #indx = [1, map(t -> max(1, min(NN, searchsortedlast(dff[:,ts.time], t))), at_times)...]

        indx = [1, searchsortedlast_both(ts.df[ts.groupingrows[a],ts.time], at_times)...]
        rand_indx = map(i -> rand(twister_arb_value_in_interval, min(indx[i+1],indx[i]+1):indx[i+1],1)[1]  ,1:(length(indx)-1))
        #dd[!,a] = dff[rand_indx, ts.value]
        dd[!,a] = ts.df[ts.groupingrows[a][rand_indx]  , ts.value]
    end
    return dd
end



# For refresh time sampling.
"""
Get refresh time after some time when all assets have an updated price.
"""
function next_tick_from_time(ts::SortedDataFrame, from_time::R; assets::Vector{Symbol} = get_assets(ts) ) where R<:Real
    ref_times = Array{R,1}()
    for a in assets
        asset_rows = ts.groupingrows[a]
        dff = ts.df[asset_rows,:]
        i = searchsortedfirst(dff[:,ts.time], from_time)
        if i >=  length(asset_rows) return missing end
        push!(ref_times,dff[i+1,ts.time])
    end
    return maximum(ref_times)
end
function next_tick(ts::SortedDataFrame, from_index::I; assets::Vector{Symbol} = get_assets(ts)) where R<:Real where I<:Integer
    inds = Array{I,1}()
    for a in assets
        ind = searchsortedfirst(ts.groupingrows[a], from_index)
        if (ind > length(ts.groupingrows[a])) return missing, -10 end
        push!(inds, ts.groupingrows[a][ind])
    end
    refresh_index = maximum(inds)
    refresh_time = ts.df[refresh_index, ts.time]
    return refresh_time, refresh_index
end



"""
Get a vector of all refresh times when all assets have an updated price.
"""
function get_all_refresh_times(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); start_time::R = minimum(ts.df[:,ts.time])) where R<:Real
    ticks = Array{R,1}()
    start_ind = searchsortedfirst(ts.df[:,ts.time], start_time) - 1
    while true
        new_tick, start_ind = next_tick(ts, start_ind + 1; assets = assets)
        if ismissing(new_tick) return ticks end
        push!(ticks, new_tick)
    end
end






# For Generalised Sampling Time
