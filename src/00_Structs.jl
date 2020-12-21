"""
    SortedDataFrame(df::DataFrame, time::Symbol = :Time, grouping::Symbol = :Name, value::Symbol = :Value)

This struct wraps a DataFrame. In the constructor function for the dataframe
we presort the data and create a mapping dict so that it is fast to subset the
DataFrame by the group.

To construct pass in the dataframe, name of time column, name of grouping column and name of value column to the constructor.
"""
struct SortedDataFrame
    df::DataFrame
    time::Symbol
    grouping::Symbol
    value::Symbol
    groupingrows::Dict{Symbol,Array{Int64,1}}
    function SortedDataFrame(df::DataFrame, time::Symbol = :Time, grouping::Symbol = :Name, value::Symbol = :Value)
       df2 = sort(df, time)
       I = Int64
       dic = Dict{Symbol,Array{I,1}}()
       for g in unique(df[:,grouping])
          dic[g] = findall(df[:,grouping] .== g)
       end
       return new(df2, time, grouping, value, dic)
    end
    function SortedDataFrame(df::DataFrame, timevar::Symbol, groupingvar::Symbol, valuevar::Symbol, dic::Dict{Symbol,Array{I,1}}) where I<:Integer
        return new(df, timevar, groupingvar, valuevar, dic)
    end
    function SortedDataFrame(df::SortedDataFrame, time::Symbol, grouping::Symbol, value::Symbol = :Value)
       return SortedDataFrame(df.df, time, grouping, value)
    end
end

"""
This returns a vector of all of the assets in the SortedDataFrame with at least some number
of observations (10 by default).
"""
function get_assets(ts::SortedDataFrame, obs_to_include::Integer = 10)
    all_assets = unique(ts.df[:,ts.grouping])
    assets = Array{Symbol,1}(undef,0)
    for a in all_assets
        if length(ts.groupingrows[a]) >= obs_to_include push!(assets, a) end
    end
    return assets
end

"""
This subsets a SortedDataFrame to only the n first observations.
"""
function subset_to_tick(ts::SortedDataFrame, n::Integer)
    newdf = ts.df[1:n,:]
    assets = get_assets(ts, 0)
    newgroupingrows = Dict{Symbol,Vector{eltype(ts.groupingrows[assets[1]])}}()
    for a in assets
        newvec = ts.groupingrows[a][ts.groupingrows[a] .<= n]
        if length(newvec) > 0 newgroupingrows[a] = newvec end
    end
    return SortedDataFrame(newdf, ts.time, ts.grouping, ts.value, newgroupingrows)
end
"""
This subsets a SortedDataFrame to only the first observations up until some time.
"""
function subset_to_time(ts::SortedDataFrame, totime::Real)
    ind = searchsortedfirst(ts.df[:,ts.time], totime) - 1
    if ind >= nrow(ts.df) return ts end
    return subset_to_tick(ts, ind)
end

"""
The time between the first and the last tick in a SortedDataFrame.
"""
function duration(ts::SortedDataFrame)
    return maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])
end

"""
This stores three elements. A Hermitian correlation matrix, a vector of volatilities
and a vector of labels. The order of the labels matches the order of the assets in
the volatility vector and correlation matrix.
"""
mutable struct CovarianceMatrix{R<:Real}
    correlation::Hermitian{R}
    volatility::Array{R,1}
    labels::Array{Symbol,1}
end



"""
This makes an empty CovarianceMatrix struct with all volatilities and correlations being NaNs
"""
function make_nan_covariance_matrix(labels::Vector{Symbol})
    d = length(labels)
    correlation = ones(d,d)
    correlation .= NaN
    correlation[diagind(correlation)] .= 1
    vols = ones(d)
    vols .= NaN
    return CovarianceMatrix(Hermitian(correlation), vols, labels)
end

"""
Calculates the mean absolute distance (elementwise) between two CovarianceMatrixs.
Undefined if any labels differ between the two CovarianceMatrixs
"""
function calculate_mean_abs_distance(cov1::CovarianceMatrix, cov2::CovarianceMatrix)
    if length(symdiff(cov1.labels, cov2.labels)) != 0 return NaN, NaN end
    N = length(cov1.labels)
    cov11 = rearrange(cov1, cov2.labels)
    cor_error = sum(abs.(cov11.correlation .- cov2.correlation)) / ((N^2-N)/2)
    vol_error = mean(abs.(cov11.volatility  .- cov2.volatility))
    return (Correlation_error = cor_error, Volatility_error = vol_error)
end

"""
Extract the correlation between two assets stored in a CovarianceMatrix
"""
function get_correlation(covar::CovarianceMatrix, asset1::Symbol, asset2::Symbol)
    index1 = findfirst(asset1 .== covar.labels)
    index2 = findfirst(asset2 .== covar.labels)
    if isnothing(index1) | isnothing(index2) return missing end
    return covar.correlation[index1, index2]
end

"""
Get the voltility for a stock from a CovarianceMatrix
"""
function get_volatility(covar::CovarianceMatrix, asset1::Symbol)
    index1 = findfirst(asset1 .== covar.labels)
    if isnothing(index1) return missing end
    return covar.volatility[index1]
end

import Statistics.mean

"""
Combine a vector of CovarianceMatrixs with equal weight on each.
"""
function mean(vec::Vector{CovarianceMatrix})
    len = length(vec)
    if len == 1 return vec[1] end
    running_cov = combine_covariance_matrices(vec[1], vec[2], 0.5, 0.5)
    if len == 2 return running_cov end
    for i in 3:len
        running_cov = combine_covariance_matrices(vec[1], vec[2], (i-1)/i, (i-1)/i)
    end
    return running_cov
end

"""
Test if a CovarianceMatrix struct contains a valid correlation matrix.
"""
function valid_correlation_matrix(mat::Hermitian)
    eig = eigen(mat).values
    if length(eig) == 0 return false end # There is no eigenvalue decomposition.
    A = minimum(eig) >= 0       # is it PSD
    B = all(abs.(diag(mat) .- 1) .< 10*eps()) # does it have a unit diagonal
    C = all(abs.(mat) .<= 1 + 10*eps())       # all all off diagonals less than one in absolute value
    return all([A,B,C])
end
valid_correlation_matrix(covar::CovarianceMatrix) = valid_correlation_matrix(covar.correlation)

"""
Count the number of observations for each asset.
"""
function ticks_per_asset(ts::SortedDataFrame, assets = get_assets(ts))
    ticks_per_asset = map(a -> length(ts.groupingrows[a]), assets)
    return Dict{Symbol,eltype(ticks_per_asset)}(assets .=> ticks_per_asset)
end


## Linear algebra extensions
import Base.+, Base.-
function +(A::Hermitian, B::Diagonal)
      return Hermitian(A .+ B)
end
+(B::Diagonal,A::Hermitian) = +(A, B)
function -(A::Hermitian, B::Diagonal)
      return Hermitian(A .- B)
end
-(B::Diagonal,A::Hermitian) = -1*(-(A, B))
