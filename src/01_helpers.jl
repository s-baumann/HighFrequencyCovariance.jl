
"""
    cov_to_cor(mat::AbstractMatrix)

Converts a matrix (representing a covariance matrix) into a `Hermitian` correlation matrix and a vector of standard deviations.
### Inputs
* `cor` - A matrix.
### Returns
* A `Hermitian`.
* A `Vector` of standard deviations (not volatilities).
"""
function cov_to_cor(mat::AbstractMatrix)
    sdevs = sqrt.(diag(mat))
    cor = mat ./ (sdevs * transpose(sdevs))
    cor[diagind(cor)] .= 1
    return Hermitian(cor), sdevs
end



"""
    cov_to_cor_and_vol(
        mat::AbstractMatrix,
        duration_of_covariance_matrix::Dates.Period,
        duration_for_desired_vols::Dates.Period,
    )

    cov_to_cor_and_vol(
        mat::AbstractMatrix,
        duration_of_covariance_matrix::Real,
        duration_for_desired_vols::Real,
    )

Converts a matrix (representing a covariance matrix) into a `Hermitian` correlation matrix and a vector of volatilities.
### Inputs
* `cor` - A correlation matrix.
* `duration_of_covariance_matrix` - The duration of the covariance matrix. If these are input as reals they must have the same units.
* `duration_for_desired_vols` - The duration you want a volatility for. If these are input as reals they must have the same units.
### Returns
* A `Hermitian`.
* A `Vector` of volatilities.

    cov_to_cor_and_vol(
        mat::AbstractMatrix,
        duration_of_covariance_matrix_in_natural_units::Real,
    )

### Inputs
* `cor` - A correlation matrix.
* `duration_of_covariance_matrix_in_natural_units` - The duration of the covariance matrix. It duration must be input in units that you know of (for instance the `time_period_per_unit` of a `SortedDataFrame`).
### Returns
* A `Hermitian`.
* A `Vector` of volatilities.
"""
function cov_to_cor_and_vol(
    mat::AbstractMatrix,
    duration_of_covariance_matrix::Dates.Period,
    duration_for_desired_vols::Dates.Period,
)
    cor, sdevs = cov_to_cor(mat)
    vols =
        sdevs /
        sqrt(time_period_ratio(duration_of_covariance_matrix, duration_for_desired_vols))
    return Hermitian(cor), vols
end
function cov_to_cor_and_vol(
    mat::AbstractMatrix,
    duration_of_covariance_matrix::Real,
    duration_for_desired_vols::Real,
)
    cor, sdevs = cov_to_cor(mat)
    vols = sdevs / sqrt(duration_of_covariance_matrix / duration_for_desired_vols)
    return Hermitian(cor), vols
end
function cov_to_cor_and_vol(
    mat::AbstractMatrix,
    duration_of_covariance_matrix_in_natural_units::Real,
)
    cor, sdevs = cov_to_cor(mat)
    vols = sdevs / sqrt(duration_of_covariance_matrix_in_natural_units)
    return Hermitian(cor), vols
end



"""
    cor_to_cov(cor::AbstractMatrix,sdevs::Vector{<:Real})

Converts a correlation matrix and some standard deviations into a `Hermitian` covariance matrix.
### Inputs
* `cor` - A correlation matrix.
* `sdevs` - A vector of standard deviations (not volatilities).
### Returns
* A `Hermitian`.
"""
function cor_to_cov(cor::AbstractMatrix, sdevs::Vector{<:Real})
    mat = cor .* (sdevs * transpose(sdevs))
    return Hermitian(mat)
end

"""
    covariance(
        cm::CovarianceMatrix,
        period::Dates.Period = cm.time_period_per_unit,
        assets::Vector{Symbol} = cm.labels,
    )

This makes a `Hermitian` matrix for the covariance matrix over some duration.
### Inputs
* `cm` - A `CovarianceMatrix` struct.
* `period` - A duration for which you want a covariance matrix. This should be in a Dates.Period.
* `assets` - What assets in include in the covariance matrix.
### Returns
* A `Hermitian`. The labelling of assets for each row/column is as per the input `assets` vector.
"""
function covariance(
    cm::CovarianceMatrix,
    period::Dates.Period = cm.time_period_per_unit,
    assets::Vector{Symbol} = cm.labels,
)
    cm2 = rearrange(cm, assets)
    sds = convert_vol(cm2.volatility, cm2.time_period_per_unit, period)
    return cor_to_cov(cm2.correlation, sds)
end

"""
    construct_matrix_from_eigen(
        eigenvalues::Vector{<:Real},
        eigenvectors::Matrix{<:Real},
    )

Constructs a matrix from its eigenvalue decomposition.
### Inputs
* `eigenvalues` - A vector of eigenvalues.
* `eigenvectors` - A matrix of eigenvectors. The i'th column corresponds to the i'th eigenvalue.
### Returns
* A `Matrix`.
"""
function construct_matrix_from_eigen(
    eigenvalues::Vector{<:Real},
    eigenvectors::Matrix{<:Real},
)
    value_matrix = zeros(length(eigenvalues), length(eigenvalues))
    value_matrix[diagind(value_matrix)] = eigenvalues
    final_mat = eigenvectors * value_matrix * transpose(eigenvectors)
    return Matrix(final_mat)
end

"""
    simple_differencing(new::Vector,old::Vector)

Does simple differencing of two vectors.
### Inputs
* `new` - A `Vector` of `Real`s.
* `old` - A `Vector` of `Real`s
### Returns
* A `Vector` of `Real`s
"""
simple_differencing(new::Vector, old::Vector) = (new .- old)

"""
    get_returns(dd::DataFrame; rescale_for_duration::Bool = false)

Converts a long format `DataFrame` of prices into a `DataFrame` of returns.
### Inputs
* `dd` - A `DataFrame` with a column called :Time and all other columns being asset prices in each period.
* `rescale_for_duration` - Should returns be rescaled to reflect a common time interval.
### Returns
* A `DataFrame` of returns.
"""
function get_returns(dd::DataFrame; rescale_for_duration::Bool = false)
    N = nrow(dd)
    assets = setdiff(Symbol.(collect(names(dd))), [:Time])
    dd_mat = Array{Float64,2}(dd[1:N, assets])
    diffs = reduce(
        hcat,
        map(
            i -> simple_differencing(dd_mat[2:end, i], dd_mat[1:(end-1), i]),
            1:length(assets),
        ),
    )
    dd_returns = diffs
    if rescale_for_duration
        time_diffs = dd[2:N, :Time] - dd[1:(N-1), :Time]
        dd_returns = (1 ./ sqrt.(time_diffs)) .* diffs
    end
    dd2 = DataFrame(dd_returns, assets)
    return dd2
end

"""
    weighted_mean(x::Vector, w::Vector)

This calculates a weighted mean given vectors for values and for weights.
    If the sum of the absolute values of the weights is close to zero a simple mean
    of x is returned instead.
### Inputs
* `x` - A `Vector` of the values you want to average.
* `w` - A `Vector` of the weights you want to use.
### Returns
* A `Real` number for the weighted mean.
"""
function weighted_mean(x::Vector, w::Vector)
    if length(x) == 0
        return NaN
    end
    if length(x) == 1
        return x[1]
    end
    # We revert to calculating a simple mean in cases where the sum of absolute
    # values of weights is less than 100 epsilons. This is to avoid issues with
    # dividing a number that is close to zero.
    if sum(abs.(w)) < 100 * eps()
        return mean(x)
    end
    return sum(x .* w) / sum(w)
end


"""
    is_missing_nan_inf(x) = (ismissing(x) || isnan(x)) || isinf(x)

This tests if a value is missing, nan or inf and returns true if one of these things is true.
### Inputs
* `x` - The object to test for missing, inf, nan.
### Returns
* A `Bool` for whether or not one of these conditions is true.
"""
is_missing_nan_inf(x) = (ismissing(x) || isnan(x)) || isinf(x)

"""
    combine_covariance_matrices(
        vect::Vector{CovarianceMatrix{T}},
        cor_weights::Vector{<:Real} = repeat([1.0], length(vect)),
        vol_weights::Vector{<:Real} = cor_weights,
        time_period_per_unit::Union{Missing,Dates.Period} = vect[1].time_period_per_unit,
    ) where T<:Real

Combines a vector of `CovarianceMatrix` structs into one `CovarianceMatrix` struct.
### Inputs
* `vect` - A vector of `CovarianceMatrix` structs.
* `cor_weights` - A vector for how much to weight the correlations from each covariance matrix (by default they will be equalweighted).
* `vol_weights` - A vector for how much to weight the volatilities from each covariance matrix (by default they will be equalweighted).
* `time_period_per_unit` - What time period should the volatilities be scaled to.
### Returns
* A matrix and a vector of labels for each row/column of the matrix.
"""
function combine_covariance_matrices(
    vect::Vector{CovarianceMatrix{T}},
    cor_weights::Vector{<:Real} = repeat([1.0], length(vect)),
    vol_weights::Vector{<:Real} = cor_weights,
    time_period_per_unit::Union{Missing,Dates.Period} = vect[1].time_period_per_unit,
) where T<:Real
    if length(vect) < 1
        error("An empty vector of covariance matrices was input. So not possible to combine.")
    end
    all_labels = union(map(x -> x.labels, vect)...)
    dims = length(all_labels)
    R = promote_type(map(x -> eltype(x.correlation), vect)...)
    new_mat = Array{R,2}(undef, dims, dims)
    new_vols = Array{R,1}(undef, dims)
    for row = 1:dims
        row_label = all_labels[row]
        for col = (row+1):dims
            col_label = all_labels[col]
            correls =
                map(i -> get_correlation(vect[i], row_label, col_label), 1:length(vect))
            valid_entries =
                setdiff(1:length(correls), findall(is_missing_nan_inf.(correls)))
            new_mat[row, col] =
                weighted_mean(correls[valid_entries], cor_weights[valid_entries])
        end
        vols = map(
            i -> convert_vol(
                get_volatility(vect[i], row_label),
                vect[i].time_period_per_unit,
                time_period_per_unit,
            ),
            1:length(vect),
        )

        valid_entries = setdiff(1:length(vols), findall(is_missing_nan_inf.(vols)))
        new_vols[row] = weighted_mean(vols[valid_entries], vol_weights[valid_entries])
    end
    new_mat[diagind(new_mat)] .= 1
    hermitian_new_mat = Hermitian(new_mat)
    return CovarianceMatrix(hermitian_new_mat, new_vols, all_labels, time_period_per_unit)
end

"""
    rearrange(
        cm::CovarianceMatrix,
        labels::Vector{Symbol},
        time_period_per_unit::Union{Missing,Dates.Period} = cm.time_period_per_unit,
    )

Rearrange the order of labels in a `CovarianceMatrix`.
### Takes
* `cm` - A `CovarianceMatrix`.
* `labels` - A `Vector` of labels.
* `time_period_per_unit` - The time period you want for the resultant Covariance Matrix
### Returns
* A `CovarianceMatrix`.
"""
function rearrange(
    cm::CovarianceMatrix,
    labels::Vector{Symbol},
    time_period_per_unit::Union{Missing,Dates.Period} = cm.time_period_per_unit,
)
    if length(setdiff(labels, cm.labels)) > 0
        error("You put in labels that are not in the CovarianceMatrix")
    end
    same_assets = (length(cm.labels) == length(labels)) && (all(cm.labels .== labels))
    same_vol = (time_period_per_unit == cm.time_period_per_unit)
    if (same_assets && same_vol)
        return cm
    end
    reordering = map(x -> findfirst(x .== cm.labels)[1], labels)
    Acor = same_assets ? cm.correlation[reordering, reordering] :
        Hermitian(cm.correlation[reordering, reordering])
    Avol = same_vol ? cm.volatility[reordering] :
        convert_vol(
        cm.volatility[reordering],
        cm.time_period_per_unit,
        time_period_per_unit,
    )
    return CovarianceMatrix(Acor, Avol, labels, time_period_per_unit)
end

"""
    squared_frobenius_distance(x1::AbstractMatrix, x2::AbstractMatrix = x1)

Returns the squared frobenius distance between two matrices. This is a real.
### Inputs
* `x1` The first matrix.
* `x2` The second matrix.
### Returns
* A Scalar.
"""
function squared_frobenius_distance(x1::AbstractMatrix, x2::AbstractMatrix = x1)
    return squared_frobenius(x1 .- x2)
end
"""
    squared_frobenius(x1::AbstractMatrix)

Returns the squared frobenius norm of a matrix. This is a real.
### Inputs
* `x1` The matrix that you want the squared frobenius norm for.
### Returns
* A Scalar.
"""
function squared_frobenius(x1::AbstractMatrix)
    p = size(x1)[1]
    return tr(x1 * transpose(x1)) / p
end



"""
    time_between_refreshes(
       ts::SortedDataFrame;
       assets::Vector{Symbol} = get_assets(ts),
    )

Get a `DataFrame` showing how many time is between each refresh and how many ticks in total.
### Inputs
* `ts` - Tick data.
* `assets` - A `Vector` of labels.
### Returns
* A `DataFrame` summarising the average number of time between ticks for each asset.
"""
function time_between_refreshes(
    ts::SortedDataFrame;
    assets::Vector{Symbol} = get_assets(ts),
)
    minn, maxx = extrema(ts.df[:, ts.time])
    total_secs = maxx - minn
    R = eltype(ts.df[:, ts.value])
    N = nrow(ts.df)
    dd = DataFrame(
        Asset = Array{Symbol,1}([]),
        time_between_ticks = Array{R,1}([]),
        number_of_ticks = Array{Int64,1}([]),
    )
    for a in assets
        asset_rows = ts.groupingrows[a]
        dff = ts.df[asset_rows, :]
        number_of_ticks = length(asset_rows)
        push!(
            dd,
            Dict(
                :Asset => a,
                :time_between_ticks => number_of_ticks / total_secs,
                :number_of_ticks => number_of_ticks,
            ),
        )
    end
    sort!(dd, :time_between_ticks)
    return dd
end
