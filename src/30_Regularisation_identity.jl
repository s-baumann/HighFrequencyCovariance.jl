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

    identity_regularisation(mat::Hermitian, asset_returns::DataFrame)

### Inputs
* mat::Hermitian - A matrix to be regularised.
* asset_returns::DataFrame - A `DataFrame` with returns for each asset.
### Returns
* A `Hermitian`.

    identity_regularisation(mat::Hermitian, asset_returns::DataFrame) where R<:Real

### Inputs
* mat::Hermitian - A matrix to be regularised.
* ts::SortedDataFrame - Tick data.
### Returns
* A `Hermitian`.

    identity_regularisation(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; spacing::Union{Missing,<:Real} = missing, apply_to_covariance::Bool = true)



This regularises the matrix by doing an elementwise convex linear combination of
it with the identity matrix. The weight the identity matrix gets is that specified
in Ledoit and Wolf. The inputs are:
* covariance_matrix::CovarianceMatrix or mat::Hermitian - The matrix to be regularised.
* asset_returns::DataFrame - A DataFrame containing returns for each asset. There should be one column for each asset.
* ts::SortedDataFrame - The tick data
* spacing::Union{Missing,<:Real} - What spacing (in time) should returns be calculated from ts. If missing refresh times will be used.
* apply_to_covariance::Bool - Should regularisation be applied to the correlation or covariance matrix.

If a `Hermitian` is input then one will be returned. If a `CovarianceMatrix` is input then one will be returned.

    identity_regularisation(mat::Hermitian, identity_weight::Real)
This regularises the matrix by doing an elementwise convex linear combination of it
 with the identity matrix (where identity_weight is the weight the identity matrix gets).
 A `Hermitian` is returned.

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
function identity_regularisation(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector;
                                 spacing::Union{Missing,<:Real} = missing)
    at_times = ismissing(spacing) ? get_all_refresh_times(ts, mat_labels) : collect(0:spacing:maximum(ts.df[:,ts.time]))
    dd_compiled = latest_value(ts, at_times; assets = mat_labels)
    asset_returns = get_returns(dd_compiled; rescale_for_duration = true)
    return identity_regularisation(mat, asset_returns)
end
function identity_regularisation(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame;
                                 spacing::Union{Missing,<:Real} = missing, apply_to_covariance::Bool = true)
     if apply_to_covariance
         regularised_covariance = identity_regularisation(covariance(covariance_matrix,1),
                                           ts, covariance_matrix.labels; spacing = spacing)
         corr, vols = cov2cor_and_vol(regularised_covariance, 1)
         return CovarianceMatrix(corr, vols, covariance_matrix.labels)
     else
         return CovarianceMatrix(Hermitian(identity_regularisation(covariance_matrix.correlation, ts, covariance_matrix.labels; spacing = spacing)), covariance_matrix.volatility, covariance_matrix.labels)
     end
end
function identity_regularisation(covariance_matrix::CovarianceMatrix, identity_weight::Real; apply_to_covariance = false)
     if apply_to_covariance
         regularised_covariance = identity_regularisation(covariance(covariance_matrix,1),identity_weight)
         corr, vols = cov2cor_and_vol(regularised_covariance, 1)
         return CovarianceMatrix(corr, vols, covariance_matrix.labels)
     else
         return CovarianceMatrix(Hermitian(identity_regularisation(covariance_matrix.correlation, identity_weight)), covariance_matrix.volatility, covariance_matrix.labels)
     end
end
