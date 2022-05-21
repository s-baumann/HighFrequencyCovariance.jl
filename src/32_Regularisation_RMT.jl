
"""
    mean_sqrt_of_positive_diagonals(x)
This returns the mean square root of the positive elements on the diagonal.
"""
function mean_sqrt_of_positive_diagonals(x)
    vals = diag(x)
    return mean(sqrt, vals[vals .> 0])^2
end

"""
Note that this is not exported with a different name to avoid confusion with
eigenvalue_clean(mat::Hermitian, eigenvalue_threshold::Real).
"""
function _eigenvalue_clean(mat::Hermitian, obs::Real)
    if sum(isnan.(mat)) > 0 return mat end # If someone inputs a matrix involving a NaN
    N = size(mat)[1]
    eigenvalues, eigenvectors = eigen(mat)
    sigma2 = 1 - maximum(eigenvalues)/N
    q = obs/N
    eigenvalue_threshold = (sigma2 * (1 + 1/q + 2*sqrt(1/q))) * mean_sqrt_of_positive_diagonals(mat)
    return eigenvalue_clean(eigenvalues, eigenvectors, eigenvalue_threshold)
end

"""
    eigenvalue_clean(eigenvalues::Vector{<:Real}, eigenvectors::Matrix{<:Real},
                     eigenvalue_threshold::Real)

This takes the small eigenvalues (with values below eigenvalue_threshold). It sets them to the
greater of their average or eigenvalue_threshold/(4*number_of_small_eigens). Then the matrix is reconstructed and returned (as a `Hermitian`)
### Inputs
* `eigenvalues` - The eigenvalues of a matrix.
* `eigenvectors` - The eigenvectors  of a matrix.
* `eigenvalue_threshold` - The threshold for a eigenvalue to be altered.
### Returns
* A `Hermitian`.


    eigenvalue_clean(mat::Hermitian, eigenvalue_threshold::Real)

This splits a matrix into its eigenvalues and eigenvectors. Then takes the small eigenvalues (with values below `eigenvalue_threshold`). It sets them to the
greater of their average or `eigenvalue_threshold/(4*number_of_small_eigens)`. Then the matrix is reconstructed and returned (as a `Hermitian`)
### Inputs
* `mat` - A matrix that you want to regularise with eigenvalue regularisation.
* `eigenvalue_threshold` - The threshold for a eigenvalue to be altered.
### Returns
* A `Hermitian`.


    eigenvalue_clean(mat::Hermitian, ts::SortedDataFrame)

Similarly to the above two methods these functions regularise a matrix by setting small eigenvalues to near zero.
The method of Laloux, Cizeau, Bouchaud & Potters 2000 is used to choose a threshold.
### Inputs
* `mat` - A matrix that you want to regularise with eigenvalue regularisation.
* `ts` - The tick data.
### Returns
* A `Hermitian`.


    eigenvalue_clean(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame;
                     apply_to_covariance::Bool = true)
### Inputs
* `mat` - A matrix that you want to regularise with eigenvalue regularisation.
* `ts` - The tick data.
* `apply_to_covariance` Should regularisation be applied to the covariance matrix or the correlation matrix.
### Returns
* A `CovarianceMatrix`.


Note that if the input matrices include any NaN terms then regularisation is not possible. The matrix will be silently returned (as these NaNs will generally be
from upstream problems so it is useful to return the matrix rather than throw at this point).As a result outputs should be checked.

## References
Laloux, L., Cizeau, P., Bouchaud J. , Potters, M. 2000. "Random matrix theory and financial correlations" International Journal of Theoretical Applied FInance, 3, 391-397.
"""
function eigenvalue_clean(eigenvalues::Vector{<:Real},
                          eigenvectors::Matrix{<:Real}, eigenvalue_threshold::Real)
    number_of_small_eigens = sum(eigenvalues .< eigenvalue_threshold)
    av_small_eigens = max(eigenvalue_threshold/(4*number_of_small_eigens) , mean(eigenvalues[eigenvalues .< eigenvalue_threshold]) )
    eigenvalues[eigenvalues .< eigenvalue_threshold] .= av_small_eigens
    regularised_mat = construct_matrix_from_eigen(eigenvalues, eigenvectors)
    return regularised_mat
end
function eigenvalue_clean(mat::Hermitian, eigenvalue_threshold::Real)
    if sum(isnan, mat) > 0 return mat end # If someone inputs a matrix involving a NaN
    eigenvalues, eigenvectors = eigen(mat)
    return eigenvalue_clean(eigenvalues, eigenvectors, eigenvalue_threshold)
end
function eigenvalue_clean(mat::Hermitian, ts::SortedDataFrame)
    dims = size(mat)[1]
    obs = nrow(ts.df)/dims
    regularised_mat = _eigenvalue_clean(mat, obs)
    return Hermitian(regularised_mat)
end
function eigenvalue_clean(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame;
                          apply_to_covariance::Bool = true)
    if apply_to_covariance
        regularised_covariance = eigenvalue_clean(covariance(covariance_matrix), ts)
        corr, vols = cov_to_cor_and_vol(regularised_covariance, covariance_matrix.time_period_per_unit, covariance_matrix.time_period_per_unit)
        return CovarianceMatrix(corr, vols, covariance_matrix.labels, covariance_matrix.time_period_per_unit)
    else
        return CovarianceMatrix(Hermitian(eigenvalue_clean(covariance_matrix.correlation, ts)), covariance_matrix.volatility, covariance_matrix.labels, covariance_matrix.time_period_per_unit)
    end
end
