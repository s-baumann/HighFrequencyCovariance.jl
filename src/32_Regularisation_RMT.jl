"""
Regularisation of the Hermitian matrix by cleaning out small eigenvalues.
### Takes
* mat - A Hermitian matrix.
* obs - How many observations were used in calculating the covariance matrix.
* eigenvalue_threshold - What threshold for deleting eigenvalues.
### Returns
* A Hermitian correlation matrix
### References
Laloux, L., Cizeau, P., Bouchaud J. , Potters, M. 2000. "Random matrix theory and financial correlations" International Journal of Theoretical Applied FInance, 3, 391-397.
"""
function eigenvalue_clean(mat::Hermitian, obs::Real; eigenvalue_threshold::Union{Missing,R} = missing) where R<:Real
    if sum(isnan.(mat)) > 0 return mat end # If someone inputs a matrix involving a NaN
    N = size(mat)[1]
    eigenvalues, eigenvectors = eigen(mat)
    #if length(eigenvalues) == 0 return mat end # If someone inputs a matrix involving a NaN
    if ismissing(eigenvalue_threshold)
        if ismissing(obs) error("Neither obs nor a threshold input. Not possible to identify eigenvalue limit") end
        sigma2 = 1 - maximum(eigenvalues)/N
        q = obs/N
        eigenvalue_threshold = (sigma2 * (1 + 1/q + 2*sqrt(1/q))) * mean_sqrt_of_positive_diagonals(mat)
    end
    number_of_small_eigens = sum(eigenvalues .< eigenvalue_threshold)
    av_small_eigens = max(eigenvalue_threshold/(4*number_of_small_eigens) , mean(eigenvalues[eigenvalues .< eigenvalue_threshold]) )
    eigenvalues[eigenvalues .< eigenvalue_threshold] .= av_small_eigens
    regularised_mat = construct_matrix_from_eigen(eigenvalues, eigenvectors)
    return regularised_mat
end

function mean_sqrt_of_positive_diagonals(x)
    vals = diag(x)
    return mean(sqrt.(vals[vals .> 0]))^2
end


"""
Regularisation of a Hermitian matrix by cleaning out small eigenvalues.
### Takes
* covariance_matrix - A CovarianceMatrix.
* ts - A SortedDataFrame
### Returns
* A CovarianceMatrix with a cleaned correlation matrix.
### References
Laloux, L., Cizeau, P., Bouchaud J. , Potters, M. 2000. "Random matrix theory and financial correlations" International Journal of Theoretical Applied FInance, 3, 391-397.
"""
function eigenvalue_clean(mat::Hermitian, ts::SortedDataFrame)
    dims = size(mat)[1]
    obs = nrow(ts.df)/dims
    regularised_mat = eigenvalue_clean(mat, obs)
    return regularised_mat
end
function eigenvalue_clean(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; apply_to_covariance::Bool = true)
    if apply_to_covariance
        regularised_covariance = eigenvalue_clean(covariance(covariance_matrix,1), ts)
        corr, vols = cov2cor_and_vol(regularised_covariance, 1)
        return CovarianceMatrix(corr, vols, covariance_matrix.labels)
    else
        return CovarianceMatrix(Hermitian(eigenvalue_clean(covariance_matrix.correlation, ts)), covariance_matrix.volatility, covariance_matrix.labels)
    end
end
