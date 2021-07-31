function put_assets_into_blocks(ts::SortedDataFrame, new_group_mult::Real)
    frequencies = time_between_refreshes(ts)
    group_stack = Vector{Vector{Symbol}}()
    this_group = Vector{Symbol}()
    this_group_start = frequencies[1,:number_of_ticks]
    for i in 1:nrow(frequencies)
        this_group_start = (length(this_group) == 0) ? frequencies[i,:number_of_ticks] : this_group_start
        if (frequencies[i,:number_of_ticks]/this_group_start - 1) > new_group_mult
            push!(group_stack, this_group) # finish off old group
            this_group_start = frequencies[i,:number_of_ticks] # Start new group
            this_group = Vector{Symbol}([frequencies[i,:Asset]])
        else
            push!(this_group, frequencies[i,:Asset])
        end
    end
    if length(this_group) > 0 push!(group_stack, this_group) end
end
#blocks = put_assets_into_blocks(ts, obs_multiple_for_new_block)

function make_adjacent_block_sequence(blocks::Vector{Vector{Symbol}})
    len = length(blocks)
    if len == 1 return [blocks...] end
    left  = make_adjacent_block_sequence(blocks[1:Integer(floor(len/2))])
    right = make_adjacent_block_sequence(setdiff(blocks[1:len], blocks[1:Integer(floor(len/2))]))
    bigblock = [(blocks...)...]
    return [bigblock, left..., right...]
end
function make_sorted_adjacent_block_sequence(blocks::Vector{Vector{Symbol}})
    unsorted = make_adjacent_block_sequence(blocks)
    lens = length.(unsorted)
    d = DataFrame(one = unsorted, lens = lens)
    sort!(d, :lens, rev=true)
    return d.one
end
#make_sorted_adjacent_block_sequence(blocks)



"""
This makes a dataframe that describes how to estimate the covariance matrix blockwise.
Hautsch, N., Kyj, L.M. and Oomen, R.C.A. (2012), A blocking and regularization approach to high‐dimensional realized covariance estimation. J. Appl. Econ., 27: 625-645
"""
function put_assets_into_blocks_by_trading_frequency(ts::SortedDataFrame, obs_multiple_for_new_block::Real, func::Symbol, optional_parameters::NamedTuple = NamedTuple())
    blocks      = put_assets_into_blocks(ts, obs_multiple_for_new_block)
    blocks2     = make_sorted_adjacent_block_sequence(blocks)
    num_blocks  = length(blocks2)
    blocking_dd = DataFrame(assets = Set.(blocks2), f = Array{Symbol}(repeat([func], num_blocks)))
    blocking_dd[!,:optional_parameters]  .= Ref(optional_parameters)

    blocking_dd[!,:number_of_assets] = map(i -> length(blocking_dd[i, :assets]), 1:nrow(blocking_dd))
    frequencies = time_between_refreshes(ts)
    blocking_dd[!,:mean_number_of_ticks] =  map(i -> mean(map(a -> frequencies[findall(frequencies[:,:Asset] .== a), :number_of_ticks], collect(blocking_dd[i, :assets])))[1], 1:nrow(blocking_dd))
    blocking_dd[!,:mean_time_per_tick]   =  map(i -> mean(map(a -> frequencies[findall(frequencies[:,:Asset] .== a), :time_between_ticks], collect(blocking_dd[i, :assets])))[1], 1:nrow(blocking_dd))
    return blocking_dd
end

"""
Run a series of covariance estimations and combine the results. Two things should be input, a SortedDataFrame with the
price update data and a dataframe describing what estimations should be performed. This should be of the same form as is
output by put_assets_into_blocks_by_trading_frequency (although the actual estimations can be customised to something different
as to what that function outputs).
"""
function blockwise_estimation(ts::SortedDataFrame, blocking_frame::DataFrame)
    covar = estimate_covariance(ts, collect(blocking_frame[1,:assets]), blocking_frame[1,:f] ; blocking_frame[1,:optional_parameters]... )
    if nrow(blocking_frame) == 1 return covar end
    for i in 2:nrow(blocking_frame)
        new_covar = estimate_covariance(ts, collect(blocking_frame[i,:assets]), blocking_frame[i,:f] ; blocking_frame[i,:optional_parameters]... )
        covar = combine_covariance_matrices([covar, new_covar], [0,1])
    end
    return covar
end
