function row_row_prime(dd, i)
    r = convert(Matrix, dd)[i,:]
    return Hermitian(r * r')
end
function b_bar(dd,S)
    N = nrow(dd)
    total = 0.0
    for k in 1:N
        total += sqrt(squared_frobenius_distance(row_row_prime.(Ref(dd), k), S))
    end
    return total / (N^2)
end

"""
Regularisation of the correlation matrix by mixing with the identity matrix as per Ledoit & Wolf 2003.

### Takes
* mat - The Hermitian matrix.
* identity_weight - How much weight to give to the identity matrix.
### Returns
* A Hermitian correlation matrix
### References
Ledoit, O. , Wolf, M. 2003. Improved Estimation of the Covariance Matrix of Stock Returns with an application to portfolio selection. Journal of empirical finance. 10. 603-621.
"""
function identity_regularisation(mat::Hermitian, identity_weight::Real)
    II = I(size(mat)[1])
    # This is a bit different to the paper as we don't have m_n. This is to ensure the diagonal stays as ones.
    mat_prime = (identity_weight .* II) + (1 - identity_weight) .* mat
    return Hermitian(mat_prime)
end
function identity_regularisation(mat::Hermitian, asset_returns::DataFrame) where R<:Real
    II   = I(size(mat)[1])
    m    = squared_frobenius_distance(mat, II) # Lemma 3.2
    d    = sqrt(squared_frobenius_distance(mat, m*II))      # Lemma 3.3
    bbar = b_bar(asset_returns, mat)
    b    = min(bbar, d)
    return identity_regularisation(mat, b/d)
end


"""
Combines the correlation matrix with the identity matrix to regularise it.
### Takes
* covariance_matrix - A CovarianceMatrix
* mat - A Hermitian Matrix
* ts - A SortedDataFrame
* identity_weight - The weight on the identity matrix.
* spacing - The spacing of returns to use in calculating the identity weight. Ignored if identity_weight provided.
* return_calc - How are returns estimated.
### Returns
* A CovarianceMatrix with a valid correlation matrix.
# References
Higham, N. J. 2001.
"""
function identity_regularisation(mat::Hermitian, mat_labels::Vector, ts::SortedDataFrame; identity_weight::Union{Missing,<:Real} = missing, spacing::Union{Missing,<:Real} = missing, return_calc::Function = simple_differencing)
    at_times = ismissing(spacing) ? get_all_refresh_times(ts, mat_labels) : collect(0:spacing:maximum(ts.df[:,ts.time]))
    dd_compiled = latest_value(ts, at_times; assets = mat_labels)
    asset_returns = get_returns(dd_compiled; rescale_for_duration = true, return_calc = return_calc)
    return identity_regularisation(mat, asset_returns)
end
function identity_regularisation(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; identity_weight::Union{Missing,<:Real} = missing,
                                 spacing::Union{Missing,<:Real} = missing, return_calc::Function = simple_differencing, apply_to_covariance::Bool = true)
     if apply_to_covariance
         regularised_covariance = identity_regularisation(covariance(covariance_matrix,1), covariance_matrix.labels, ts; identity_weight = identity_weight, spacing = spacing, return_calc = return_calc)
         corr, vols = cov2cor_and_vol(regularised_covariance, 1)
         return CovarianceMatrix(corr, vols, covariance_matrix.labels)
     else
         return CovarianceMatrix(Hermitian(identity_regularisation(covariance_matrix.correlation, covariance_matrix.labels, ts; identity_weight = identity_weight, spacing = spacing, return_calc = return_calc)), covariance_matrix.volatility, covariance_matrix.labels)
     end
end
