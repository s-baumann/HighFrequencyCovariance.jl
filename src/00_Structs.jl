"""
    SortedDataFrame(df::DataFrame, time::Symbol = :Time, grouping::Symbol = :Name, value::Symbol = :Value)

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
struct SortedDataFrame
    df::DataFrame
    time::Symbol
    grouping::Symbol
    value::Symbol
    groupingrows::Dict{Symbol,Array{Int64,1}}
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
       return new(df2, time, grouping, value, dic, time_period_per_unit)
    end
    function SortedDataFrame(df::DataFrame, timevar::Symbol, groupingvar::Symbol, valuevar::Symbol,
                             dic::Dict{Symbol,Vector{I}}, vol_unit::Dates.Period) where I<:Integer
        return new(df, timevar, groupingvar, valuevar, dic, vol_unit)
    end
end

function time_period_ratio(neww::Dates.Period, oldd::Dates.Period)
    return Nanosecond(neww) / Nanosecond(oldd)
end
function safe_multiply_period(scalar::Real, neww::Dates.Period)
    vall = Nanosecond(neww).value * scalar
    return Nanosecond(floor(vall))
end

import Base.show, Base.print

"""
    show(sdf::SortedDataFrame, number_of_rows = 10)
Show a SortedDataFrame with a set number of rows.
### Inputs
* `sdf` - The `SortedDataFrame` to show.
* `number_of_rows` - The number of rows to show.
"""
function show(sdf::SortedDataFrame, number_of_rows = 10)
    println()
    println("SortedDataFrame with " , nrow(sdf.df), " rows."  )
    show(stdout, sdf.df[1:number_of_rows,[sdf.time, sdf.grouping, sdf.value]] ; summary = false)
    println()
end
"""
    print(sdf::SortedDataFrame, number_of_rows = 10)
Show a SortedDataFrame with a set number of rows.
### Inputs
* `sdf` - The `SortedDataFrame` to print.
* `number_of_rows` - The number of rows to print.
"""
function print(sdf::SortedDataFrame, number_of_rows = 10)
    show(sdf, number_of_rows)
end

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
        if sum(isnan.(ts.df[ts.groupingrows[a], ts.value])) > 0
            @warn string("There are nan values for ", a)
        end
        vals = ts.df[ts.groupingrows[a], ts.value]
        cond2 = (maximum(vals) - minimum(vals) > 1000*eps())
        if (cond1 & cond2) push!(assets, a) end
    end
    return sort(assets)
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

"""
    CovarianceMatrix(correlation::Hermitian{R}, volatility::Vector{R},
                     labels::Vector{Symbol}) where R<:Real

This `Struct` stores three elements. A `Hermitian` correlation matrix, a vector of volatilities
and a vector of labels. The order of the labels matches the order of the assets in
the volatility vector and correlation matrix.
The default constructor is used.
### Inputs
* `correlation` - A `Hermitian` correlation matrix.
* `volatility` - Volatilities for each asset.
* `labels` - The labels for the `correlation` and `volatility` members. The n'th entry of the `labels` vector should contain the name of the asset that has its volatility in the n'th entry of the `volatility` member and its correlations in the n'th row/column of the `correlation` member.
* `time_period_per_unit` - The period that one unit of volatility corresponds to.
### Returns
* A `CovarianceMatrix`.
"""
mutable struct CovarianceMatrix{R<:Real}
    correlation::Hermitian{R}
    volatility::Vector{R}
    labels::Vector{Symbol}
    time_period_per_unit::Dates.Period
end

"""
    show(cm::CovarianceMatrix)
    show(cm::CovarianceMatrix, decimal_places_volatility::Integer, decimal_places_correlation::Integer)

This prints the `CovarianceMatrix` in a nice format.
### Inputs
* `cm` - The `CovarianceMatrix` you want to print.
* `decimal_places_volatility`  - The number of digits you want to show for volatilities. If not input then all digits are shown.
* `decimal_places_correlation` - The number of digits you want to show. If not input then all digits are shown.
"""
function show(cm::CovarianceMatrix)
    println()
    println("Volatilities per time interval of ", cm.time_period_per_unit)
    flat_labels = reshape( cm.labels , (1, length(cm.labels)))
    Base.print_matrix(stdout, vcat(flat_labels, cm.volatility'))
    println("\n")
    println("Correlations")
    corr = cm.correlation
    corr = vcat(flat_labels, corr)
    corr = hcat([:___, cm.labels...], corr)
    Base.print_matrix(stdout, corr)
    println("\n")
end
function show(cm::CovarianceMatrix, sig_figs_volatility::Integer, decimal_places_correlation::Integer)
    cm2 = deepcopy(cm)
    cm2.volatility  = round.(cm2.volatility , sigdigits = sig_figs_volatility)
    cm2.correlation = Hermitian(round.(cm2.correlation, digits = decimal_places_correlation))
    show(cm2)
end

"""
    convert_vol(vol::Union{Vector{<:Real},Real}, vol_period::Dates.Period, new_vol_period::Dates.Period)

### Inputs
* `Vol` - The volatility you want to convert.
* `vol_period` - The period of the volatility you have.
* `new_vol_period` - The period that you want to convert the volatility to.
"""
function convert_vol(vol::Union{Vector{<:Real},Real}, vol_period::Dates.Period, new_vol_period::Dates.Period)
    return vol .* sqrt( time_period_ratio(new_vol_period, vol_period) )
end
convert_vol(vol::Missing, vol_period::Dates.Period, new_vol_period::Dates.Period) = missing

"""
    print(cm::CovarianceMatrix)

This prints the `CovarianceMatrix` in a nice format.
### Inputs
* `cm` - The `CovarianceMatrix` you want to print.
* `decimal_places_volatility`  - The number of digits you want to show for volatilities. If not input then all digits are shown.
* `decimal_places_correlation` - The number of digits you want to show. If not input then all digits are shown.
"""
function print(cm::CovarianceMatrix)
    show(cm)
end
function print(cm::CovarianceMatrix, decimal_places_volatility::Integer, decimal_places_correlation::Integer)
    show(cm, decimal_places_volatility, decimal_places_correlation)
end

"""
    make_nan_covariance_matrix(labels::Vector{Symbol}, time_period_per_unit::Dates.Period)

This makes an empty `CovarianceMatrix` struct with all volatilities and correlations being NaNs.
### Inputs
* `labels` - The names of the asset names for this (empty) `CovarianceMatrix`.
* `time_period_per_unit` - The time interval the volatilities will be for.
### Returns
* An (empty) `CovarianceMatrix`
"""
function make_nan_covariance_matrix(labels::Vector{Symbol}, time_period_per_unit::Dates.Period)
    d = length(labels)
    correlation = ones(d,d)
    correlation .= NaN
    correlation[diagind(correlation)] .= 1
    vols = ones(d)
    vols .= NaN
    return CovarianceMatrix(Hermitian(correlation), vols, labels, time_period_per_unit)
end

"""
    calculate_mean_abs_distance(cov1::CovarianceMatrix, cov2::CovarianceMatrix, decimal_places::Integer = 8; return_nans_if_symbols_dont_match::Bool = true)

Calculates the mean absolute distance (elementwise in L1 norm) between two `CovarianceMatrix`s.
Undefined if any labels differ between the two `CovarianceMatrix`s.
### Inputs
* `cov1` - The first `CovarianceMatrix`
* `cov2` - The second `CovarianceMatrix`
* `decimal_places` - How many decimal places to show the result to.
* `return_nans_if_symbols_dont_match` - If the symbols don't match should it be an error. Or if false we only compare common symbols in both `CovarianceMatrix`s
### Returns
* An `Tuple` with the distance for correlations in first entry and distance for volatilities in the second.


    calculate_mean_abs_distance(d1::Dict{Symbol,<:Real}, d2::Dict{Symbol,<:Real})

Calculates the mean absolute distance (elementwise in L1 norm) between two `CovarianceMatrix`s.
### Inputs
* `d1` - The first `Dict`
* `d2` - The second `Dict`
### Returns
* A scalar with the mean distance between matching elements.
"""
function calculate_mean_abs_distance(cov1::CovarianceMatrix, cov2::CovarianceMatrix, decimal_places::Integer = 8; return_nans_if_symbols_dont_match::Bool = true)
    if return_nans_if_symbols_dont_match & (length(symdiff(cov1.labels, cov2.labels)) != 0) return (Correlation_error = NaN, Volatility_error = NaN) end
    labels = intersect(cov1.labels, cov2.labels)
    N = length(labels)
    cov11 = rearrange(cov1, labels)
    cov22 = rearrange(cov2, labels)
    cor_error = round(sum(abs.(cov11.correlation .- cov22.correlation)) / ((N^2-N)/2), digits = decimal_places)
    vol_error = round(mean(abs.(cov11.volatility  .- cov22.volatility)), digits  = decimal_places)
    return (Correlation_error = cor_error, Volatility_error = vol_error)
end
function calculate_mean_abs_distance(d1::Dict{Symbol,<:Real}, d2::Dict{Symbol,<:Real})
    dist = 0.0
    common_labels = intersect(collect(keys(d1)), collect(keys(d2)))
    for label in common_labels
        dist += abs(d1[label] - d2[label])
    end
    return dist / length(common_labels)
end

"""
    calculate_mean_abs_distance_covar(cov1::CovarianceMatrix, cov2::CovarianceMatrix, decimal_places::Integer = 8; return_nans_if_symbols_dont_match::Bool = true)

Calculates the mean absolute distance (elementwise in L1 norm) between the covariance matrices (over the natural time unit) of two `CovarianceMatrix`s.
Undefined if any labels differ between the two `CovarianceMatrix`s.

Note that this is different from calculate_mean_abs_distance in that this function returns one real for the distance between actual covariance matrices
rather than a tuple showing the distances in terms of correlation and volatility.

### Inputs
* `cov1` - The first `CovarianceMatrix`
* `cov2` - The second `CovarianceMatrix`
* `decimal_places` - How many decimal places to show the result to.
* `return_nans_if_symbols_dont_match` - If the symbols don't match should it be an error. Or if false we only compare common symbols in both `CovarianceMatrix`s
### Returns
* A Real
"""
function calculate_mean_abs_distance_covar(cov1::CovarianceMatrix, cov2::CovarianceMatrix, decimal_places::Integer = 8; return_nans_if_symbols_dont_match::Bool = true)
    if return_nans_if_symbols_dont_match & (length(symdiff(cov1.labels, cov2.labels)) != 0) return (Correlation_error = NaN, Volatility_error = NaN) end
    labels = intersect(cov1.labels, cov2.labels)
    N = length(labels)
    cov11 = rearrange(cov1, labels)
    cov22 = rearrange(cov2, labels)
    covar1 = covariance(cov11, cov1.time_period_per_unit)
    covar2 = covariance(cov22, cov2.time_period_per_unit)
    error = round(mean(abs.(covar1 .- covar2)), digits = decimal_places)
    return error
end


"""
    get_correlation(covar::CovarianceMatrix, asset1::Symbol, asset2::Symbol)

Extract the correlation between two assets stored in a CovarianceMatrix.
### Inputs
* `covar` - A `CovarianceMatrix`
* `asset1` - A `Symbol` representing an asset.
* `asset2` - A `Symbol` representing an asset.
### Returns
* A Scalar (the correlation coefficient).
"""
function get_correlation(covar::CovarianceMatrix, asset1::Symbol, asset2::Symbol)
    index1 = findfirst(asset1 .== covar.labels)
    index2 = findfirst(asset2 .== covar.labels)
    if isnothing(index1) | isnothing(index2) return missing end
    return covar.correlation[index1, index2]
end

"""
    get_volatility(covar::CovarianceMatrix, asset1::Symbol, time_period_per_unit::Dates.Period = Second(1))

Get the volatility for a stock from a `CovarianceMatrix`.
### Inputs
* `covar` - A `CovarianceMatrix`
* `asset1` - A `Symbol` representing an asset.
* `time_period_per_unit` - The time interval the volatilities will be for.
### Returns
* A Scalar (the volatility).
"""
function get_volatility(covar::CovarianceMatrix, asset1::Symbol, time_period_per_unit::Dates.Period = covar.time_period_per_unit)
    index1 = findfirst(asset1 .== covar.labels)
    if isnothing(index1) return missing end
    return convert_vol(covar.volatility[index1], covar.time_period_per_unit, time_period_per_unit)
end

"""
    is_psd_matrix(mat::Hermitian, min_eigen_threshold::Real = 0.0)
    is_psd_matrix(covar::CovarianceMatrix, min_eigen_threshold::Real = 0.0)

Test if a matrix is psd (Positive Semi-Definite). This is done by seeing if all eigenvalues are positive.
If a `Hermitian` is input then it will be tested. If a `CovarianceMatrix` is input then its correlation matrix will be tested.
### Inputs
* `mat` - A `Hermitian` matrix or a `CovarianceMatrix`
* `min_eigen_threshold` - How big does the smallest eigenvalue have to be.
### Returns
* A `Bool` that is true if mat is psd and false if not.
"""
function is_psd_matrix(mat::Hermitian, min_eigen_threshold::Real = 0.0)
    eig = eigen(mat).values
    if length(eig) == 0 return false end # There is no eigenvalue decomposition.
    return minimum(eig) > 0 # is it PSD
end
is_psd_matrix(covar::CovarianceMatrix) = is_psd_matrix(covar.correlation)

"""
    valid_correlation_matrix(mat::Hermitian)
    valid_correlation_matrix(covar::CovarianceMatrix)

Test if a `Hermitian` matrix is a valid correlation matrix. This is done by testing if it is psd, if it has a unit diagonal and if all other elements are less than one.
If a `Hermitian` is input then it will be tested. If a `CovarianceMatrix` is input then its correlation matrix will be tested.
### Inputs
* `mat` - A `Hermitian` matrix or a `CovarianceMatrix`
* `min_eigen_threshold` - How big does the smallest eigenvalue have to be.
### Returns
* A `Bool` that is true if mat is a valid correlation matrix and false if not.
"""
function valid_correlation_matrix(mat::Hermitian, min_eigen_threshold::Real = 0.0)
    A = is_psd_matrix(mat, min_eigen_threshold)
    B = all(abs.(diag(mat) .- 1) .< 10*eps()) # does it have a unit diagonal
    C = all(abs.(mat) .<= 1 + 10*eps())       # all all off diagonals less than one in absolute value
    return all([A,B,C])
end
valid_correlation_matrix(covar::CovarianceMatrix, min_eigen_threshold::Real = 0.0) = valid_correlation_matrix(covar.correlation, min_eigen_threshold)

"""
    ticks_per_asset(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts))

Count the number of observations for each asset.
### Inputs
* `ts` - The tick data
* `assets` - A vector with asset `Symbol`s.
### Returns
* A `Dict` with the number of observations for each input asset.
"""
function ticks_per_asset(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts))
    ticks_per_asset = map(a -> length(ts.groupingrows[a]), assets)
    return Dict{Symbol,eltype(ticks_per_asset)}(assets .=> ticks_per_asset)
end
