
"""
Converts a matrix (representing a covariance matrix) into a Hermitian correlation
matrix and a vector of standard deviations.
### Takes
* cor::AbstractMatrix - A matrix.
### Returns
* A `Hermitian`.
* A `Vector` of standard deviations (not volatilities).
"""
function cov2cor(mat::AbstractMatrix)
    sdevs = sqrt.(diag(mat))
    cor = mat ./ (sdevs * transpose(sdevs))
    cor[diagind(cor)] .= 1
    return Hermitian(cor), sdevs
end
"""
Converts a matrix (representing a covariance matrix) into a Hermitian correlation
matrix and a vector of volatilities.
### Takes
* cor::AbstractMatrix - A correlation matrix.
* duration::Real - The duration that the covariance matrix is for.
### Returns
* A `Hermitian`.
* A `Vector` of volatilities.
"""
function cov2cor_and_vol(mat::AbstractMatrix, duration::Real = 1)
    cor, sdevs = cov2cor(mat)
    return Hermitian(cor), sdevs/sqrt(duration)
end

"""
Converts a correlation matrix and some standard deviations into a Hermitian covariance matrix.
### Takes
* cor::AbstractMatrix - A correlation matrix.
* sdevs::Vector{<:Real} - A vector of standard deviations (not volatilities - use
                       sdevs = sqrt(duration) .* volatilities to convert if necessary).
### Returns
* A `Hermitian`.
"""
function cor2cov(cor::AbstractMatrix,sdevs::Vector{<:Real})
    mat = cor .* (sdevs * transpose(sdevs))
    return Hermitian(mat)
end

"""
This makes a Hermitian matrix for the covariance matrix over some duration.
### Takes
* cm::CovarianceMatrix - A CovarianceMatrix struct.
* duration::Real - A duration. This should be in same units as used in estimating cm's volatilities.
### Returns
* A `Hermitian`.
"""
function covariance(cm::CovarianceMatrix, duration::Real)
    sds = sqrt.((cm.volatility.^2) .* duration)
    return cor2cov(cm.correlation, sds)
end

"""
Constructs a matrix from its eigenvalue decomposition.
### Takes
* eigenvalues::Vector{<:Real} - A vector of eigenvalues.
* eigenvectors::Matrix{<:Real} - A matrix of eigenvectors. The i'th column corresponds to the i'th eigenvalue.
### Returns
* A `Matrix`.
"""
function construct_matrix_from_eigen(eigenvalues::Vector{<:Real}, eigenvectors::Matrix{<:Real})
    value_matrix = zeros(length(eigenvalues), length(eigenvalues))
    value_matrix[diagind(value_matrix)] = eigenvalues
    final_mat = eigenvectors * value_matrix * transpose(eigenvectors)
    return final_mat
end

"""
Does simple differencing of two vectors.
### Takes
* new::Vector - A `Vector` of `Real`s.
* old::Vector - A `Vector` of `Real`s
### Returns
* A `Vector` of `Real`s
"""
simple_differencing(new::Vector,old::Vector) =  (new .- old)

"""
Converts A long format dataframe of prices into a dataframe of returns.
### Takes
* dd::DataFrame - A `DataFrame` with a column called :Time and all other columns being asset prices in each period.
* rescale_for_duration - Should returns be rescaled.
### Returns
* A DataFrame of returns.
"""
function get_returns(dd::DataFrame; rescale_for_duration::Bool = false)
    N = nrow(dd)
    assets = setdiff(Symbol.(collect(names(dd))), [:Time])
    dd_mat = Array{Float64,2}(dd[1:N,assets])
    time_diffs = dd[2:N,:Time] - dd[1:(N-1),:Time]
    diffs  = reduce(hcat, map(i -> simple_differencing(dd_mat[2:end,i] , dd_mat[1:(end-1),i]) , 1:length(assets)) )
    dd_returns = rescale_for_duration ? (1 ./ sqrt.(time_diffs)) .*  diffs : diffs
    dd2 = DataFrame(dd_returns, assets)
    return dd2
end

function weighted_mean(x::Vector, w::Vector)
    if length(x) == 0 return NaN end
    if length(x) == 1 return x[1] end
    if sum(abs.(w)) < 100*eps() return mean(x) end
    return sum(x .* w) / sum(w)
end

is_missing_nan_inf(x) = (ismissing(x) | isnan(x)) | isinf(x)

"""
Combines a vector of CovarianceMatrix structs into one CovarianceMatrix struct.
### Inputs
* vect::Vector{CovarianceMatrix{<:Real}} - A vector of of CovarianceMatrix structs.
* cor_weights - A vector for how much to weight the correlations from each covariance matrix (by default they will be equalweighted).
* vol_weights - A vector for how much to weight the volatilities from each covariance matrix (by default they will be equalweighted).
### Returns
* A matrix (Array{Union{Missing,R},1} where R<:Real) and a vector of labels for each row/column of the matrix.
"""
function combine_covariance_matrices(vect::Vector{CovarianceMatrix{REAL}}, cor_weights::Vector{<:Real} = repeat([1.0], length(vect)), vol_weights::Vector{<:Real} = cor_weights) where REAL<:Real
    all_labels = union(map(x -> x.labels, vect)...)
    dims = length(all_labels)
    R = promote_type(map(x -> eltype(x.correlation), vect)...)
    new_mat  = Array{R,2}(undef, dims, dims)
    new_vols = Array{R,1}(undef, dims)
    for row in 1:dims
        row_label = all_labels[row]
        for col in (row+1):dims
            col_label = all_labels[col]
            correls = map(i -> get_correlation(vect[i], row_label, col_label),  1:length(vect))
            valid_entries = setdiff(1:length(correls), findall(is_missing_nan_inf.(correls)))
            new_mat[row,col] = weighted_mean(correls[valid_entries], cor_weights[valid_entries])
        end
        vols = map(i -> get_volatility(vect[i], row_label),  1:length(vect))
        valid_entries = setdiff(1:length(vols), findall(is_missing_nan_inf.(vols)))
        new_vols[row] = weighted_mean(vols[valid_entries], vol_weights[valid_entries])
    end
    new_mat[diagind(new_mat)] .= 1
    hermitian_new_mat = Hermitian(new_mat)
    return CovarianceMatrix(hermitian_new_mat, new_vols, all_labels)
end

"""
Rearrange the order of labels in a CovarianceMatrix
### Takes
* cm::CovarianceMatrix - A `CovarianceMatrix`.
* labels::Vector{Symbol} - A `Vector` of labels.
### Returns
* A `CovarianceMatrix`.
"""
function rearrange(cm::CovarianceMatrix, labels::Vector{Symbol})
  if length(symdiff(labels, cm.labels)) > 0 error("You have either put in labels that are not in the covariance matrix or you have not put in all the labels that are in the covariance matrix") end
  reordering = map(x -> findfirst(x .== cm.labels)[1], labels)
  Acor = Hermitian(cm.correlation[reordering,reordering])
  Avol = cm.volatility[reordering]
  return CovarianceMatrix(Acor, Avol, labels)
end

"""
Rearrange the squared frobenius distance between two matrices. Returns a real
"""
function squared_frobenius_distance(x1::AbstractMatrix, x2::AbstractMatrix = x1)
    return squared_frobenius(x1 .- x2)
end
"""
Rearrange the squared frobenius norm of a matrix. Returns a real.
"""
function squared_frobenius(x1::AbstractMatrix)
    p = size(x1)[1]
    return tr(x1 * transpose(x1))/p
end



"""
Get a DataFrame showing how many time is between each refresh and how many ticks in total.
### Takes
* ts::SortedDataFrame - Tick data.
* assets::Vector{Symbol} - A `Vector` of labels.
### Returns
* A `DataFrame` summarising the average number of time between ticks for each asset.
"""
function time_between_refreshes(ts::SortedDataFrame; assets::Vector{Symbol} = get_assets(ts))
    total_secs = maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])
    R = eltype(ts.df[:,ts.value])
    N = nrow(ts.df)
    dd = DataFrame(Asset = Array{Symbol,1}([]), time_between_ticks = Array{R,1}([]), number_of_ticks = Array{Int64,1}([]))
    for a in assets
        asset_rows = ts.groupingrows[a]
        dff = ts.df[asset_rows,:]
        number_of_ticks = length(asset_rows)
        push!(dd, Dict(:Asset => a, :time_between_ticks => number_of_ticks/total_secs, :number_of_ticks => number_of_ticks))
    end
    sort!(dd, :time_between_ticks)
    return dd
end
