
"""
    row_row_prime(dd, i, type::Type = eltype(dd[1,1]))
This multiplies row i of dataframe dd by its inverse. Then returns that as a Hermitian.
### Inputs
* `dd` - A dataframe
* `i` - The row of the dataframe
* `type` - The numeric type of the dataframe.
### Returns
* A `Hermitian`.
"""
function row_row_prime(dd, i, type::Type = eltype(dd[1,1]))
    r = Array{type}(dd[i,:])
    return Hermitian(r * r')
end

"""
    b_bar(dd,S)
This calculates the \bar{b} expression for the Ledoit-Wolf paper
"""
function b_bar(dd,S)
    etype = eltype(dd[1,1])
    N = nrow(dd)
    total = 0.0
    for k in 1:N
        total += sqrt(squared_frobenius_distance(row_row_prime.(Ref(dd), k, etype), S))
    end
    return total / (N^2)
end

"""
    identity_regularisation(mat::Hermitian, identity_weight::Real)

Regularisation of the correlation matrix by mixing with the identity matrix.
### Inputs
* `mat` - A matrix to be regularised.
* `identity_weight` - How much weight to give to the identity matrix. Should be between 0 and 1.
### Returns
* A `Hermitian`.


    identity_regularisation(mat::Hermitian, asset_returns::DataFrame) where R<:Real

Regularisation of the correlation matrix by mixing with the identity matrix as per Ledoit & Wolf 2003.
### Inputs
* `mat` - A matrix to be regularised.
* `ts` - Tick data.
### Returns
* A `Hermitian`.


    identity_regularisation(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector;
                            spacing::Union{Missing,<:Real} = missing)

Regularisation of the correlation matrix by mixing with the identity matrix as per Ledoit & Wolf 2003.
### Inputs
* `mat` - A matrix to be regularised.
* `ts` - Tick data.
* `mat_labels` - The labels for each asset in the matrix.
* `spacing` A spacing to use to estimate returns. This is used in determining the optimal weight to give to the identity matrix.
### Returns
* A `Hermitian`.


    identity_regularisation(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame;
                            spacing::Union{Missing,<:Real} = missing, apply_to_covariance::Bool = true)

Regularisation of the correlation matrix by mixing with the identity matrix as per Ledoit & Wolf 2003.
### Inputs
* `covariance_matrix` - The `CovarianceMatrix` to be regularised.
* `ts` - Tick data.
* `spacing` A spacing to use to estimate returns. This is used in determining the optimal weight to give to the identity matrix.
* `apply_to_covariance` Should regularisation be applied to the covariance matrix or the correlation matrix.
### Returns
* A `CovarianceMatrix`.

    identity_regularisation(covariance_matrix::CovarianceMatrix, identity_weight::Real;
                            apply_to_covariance = false)

Regularisation of the correlation matrix by mixing with the identity matrix.
### Inputs
* `covariance_matrix` - The `CovarianceMatrix` to be regularised.
* `identity_weight` - How much weight to give to the identity matrix. Should be between 0 and 1.
* `apply_to_covariance` Should regularisation be applied to the covariance matrix or the correlation matrix.
### Returns
* A `CovarianceMatrix`.

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
         actual_covariance = covariance(covariance_matrix)
         regularised_covariance = identity_regularisation(actual_covariance,
                                           ts, covariance_matrix.labels; spacing = spacing)
         corr, vols = cov_to_cor_and_vol(regularised_covariance, 1)
         return CovarianceMatrix(corr, vols, covariance_matrix.labels, covariance_matrix.time_period_per_unit)
     else
         return CovarianceMatrix(Hermitian(identity_regularisation(covariance_matrix.correlation, ts, covariance_matrix.labels; spacing = spacing)),
                     covariance_matrix.volatility, covariance_matrix.labels, covariance_matrix.time_period_per_unit)
     end
end
function identity_regularisation(covariance_matrix::CovarianceMatrix, identity_weight::Real; apply_to_covariance = false)
     if apply_to_covariance
         regularised_covariance = identity_regularisation(covariance(covariance_matrix,1),identity_weight)
         corr, vols = cov_to_cor_and_vol(regularised_covariance, 1)
         return CovarianceMatrix(corr, vols, covariance_matrix.labels, covariance_matrix.time_period_per_unit)
     else
         return CovarianceMatrix(Hermitian(identity_regularisation(covariance_matrix.correlation, identity_weight)),
                       covariance_matrix.volatility, covariance_matrix.labels, covariance_matrix.time_period_per_unit)
     end
end
