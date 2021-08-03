function set_diagonal_to_one(A::Diagonal)
      A[diagind(A)] .= 1
      return Diagonal(A)
end
function set_diagonal_to_one(A::Hermitian)
      A[diagind(A)] .= 1
      return Hermitian(A)
end

function sqrt_psd(A::Hermitian)
    eigenvalues, eigenvectors = eigen(A)
    return Hermitian(construct_matrix_from_eigen(sqrt.(eigenvalues), eigenvectors))
end
sqrt_psd(A::Diagonal) = sqrt(A)

"""
This maps the Hermitian/Hermitian matrix A to the nearest matrix in the U space (the space of all unit diagonal matrices
as defined by Higham 2001). The inverse weight matrix invW determines how much to adjust
each element to get it to be unit diagonal. The weight matrix must be Hermitian positive definite.
We use the W-norm (as defined by Higham 2001).
# References
Higham, N. J. 2001. Bottom of page 335.
"""
function project_to_U(A::Union{Diagonal,Hermitian}, invW::Hermitian)
      rhs = diag(A - Diagonal(I(size(A)[1])))
      WmWm_hadamard = Hermitian(invW .^ 2)
      theta = WmWm_hadamard \ rhs
      diagtheta = Diagonal(theta)
      newA = A - Hermitian(invW * diagtheta * invW)
    return newA
end
function project_to_U(A::Union{Diagonal,Hermitian}, invW::Diagonal)
      newA = set_diagonal_to_one(A)
      return newA
end

"""
This maps a matrix to the nearest psd matrix. W_root should be the principal square root of a psd Hermitian weighting matrix, W.
W_inv_sqrt should be the corresponding square root of the inverse of W.
Higham, N. J. 2001. Theorem 3.2

`nearest_psd_matrix` is a simpler interface for this function however it does not allow weighting matrices to be specified.
"""
function project_to_S(A::Hermitian, W_root::Union{Hermitian,Diagonal}; W_inv_sqrt::Union{Hermitian,Diagonal} = sqrt_psd(inv(W_root^2)))
      Wroot_A_WRoot = Hermitian(W_root * A * W_root)
      eigenvalues, eigenvectors = eigen(Wroot_A_WRoot)
      if all(eigenvalues .> 0) return A end
      positive_eigenvalues = map(x-> max(0,x), eigenvalues)
      Wroot_A_WRoot_positiveEigen = W_inv_sqrt * construct_matrix_from_eigen(positive_eigenvalues, eigenvectors) * W_inv_sqrt
      return Hermitian(Wroot_A_WRoot_positiveEigen)
end
project_to_S(A::Diagonal, W_root::Union{Hermitian,Diagonal}; W_inv_sqrt::Union{Hermitian,Diagonal,Missing} = missing) = A


"""
Do one iterate mapping the input matrix to the S space (of psd matrices) and then to
the U space (unit diagonal and all other entries below 1 in absolute value).
Returns the updated matrix and the next iterate's Dykstra correction.
Higham, N. J. 2001. Algorithm 3.3
"""
function iterate_higham(Y::Union{Hermitian,Diagonal}, Dykstra::Union{Hermitian,Diagonal}, W_root::Union{Hermitian,Diagonal}, W_inv::Union{Hermitian,Diagonal}, W_inv_sqrt::Union{Hermitian,Diagonal})
    R = Y - Dykstra
    X = project_to_S(R, W_root; W_inv_sqrt = W_inv_sqrt)
    new_Dykstra = X - R
    new_Y = project_to_U(X, W_inv)
    return new_Y, new_Dykstra
end

"""
    nearest_correlation_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                                 doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    nearest_correlation_matrix(covariance_matrix::CovarianceMatrix; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                                 doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    nearest_correlation_matrix(mat::Hermitian, ts::SortedDataFrame, mat_labels = missing; weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
                                 doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    nearest_correlation_matrix(mat::Hermitian, mat_labels::Vector = missing; weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
                                 doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)

Maps a matrix to the nearest valid correlation matrix (pdf matrix with unit diagonal and all other entries below 1 in absolute value).


These functions calls the `iterate_higham` function to move a matrix towards it nearest correlation matrix until it hits a fixed point.
 * covariance_matrix::CovarianceMatrix or mat::Hermitian - The matrix to be regularised.
 * ts::SortedDataFrame - The tick data
 * weighting_matrix::Union{Diagonal,Hermitian} - What weighting matrix should be used (in determining what is the nearest correlation matrix).
 * doDykstra::Bool Should a Dykstra correction be done.
 * stop_at_first_correlation_matrix::Bool  Should the iteration stop at the first valid correlation matrix or continue until all iterates have been performed.
 * max_iterates::Integer What is the maximum number of iterates to do. If stop_at_first_correlation_matrix = false then max_iterates is the number of iterates that we will do.
If a `Hermitian` is input then one will be returned. If a `CovarianceMatrix` is input then one will be returned.


     nearest_correlation_matrix(mat::Union{Diagonal,Hermitian}, W::Union{Diagonal,Hermitian} = Diagonal(Float64.(I(size(mat)[1])));
                                 doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
This also iterates a matrix towards convergence but returns a tuple with the updated
 Hermitian/Diagonal matrix in the first slot, the number of iterates in the second
 and a Symbol representing a convergence status in the third. The arguments are as above.
"""
function nearest_correlation_matrix(mat::AbstractMatrix, W::Union{Diagonal,Hermitian} = Diagonal(Float64.(I(size(mat)[1])));
                                    doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    @assert all(size(mat) .== size(W))
    W_root = sqrt_psd(W)
    W_inv = inv(W)
    W_inv_sqrt = sqrt_psd(W_inv)

    N = size(mat)[1]
    Y = mat
    ZeroDykstra = Hermitian(zeros(N,N))
    Dykstra = ZeroDykstra
    counter = 1
    while counter < max_iterates
        Y, Dykstra = doDykstra ? iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt) : iterate_higham(Y, ZeroDykstra, W_root, W_inv, W_inv_sqrt)
        if stop_at_first_correlation_matrix
             valid_correlation_matrix
             spd = valid_correlation_matrix(Y)
             if spd return Y, counter, :already_valid_correlation_matrix end
        end
        counter += 1
    end
    return (updated_matrix = Y, counter = counter, convergence = :reached_maxiter)
end
function nearest_correlation_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    return nearest_correlation_matrix(covariance_matrix; weighting_matrix = weighting_matrix,
                                 doDykstra = doDykstra, stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
end
function nearest_correlation_matrix(covariance_matrix::CovarianceMatrix; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    regularised_correl, counter, convergence = nearest_correlation_matrix(covariance_matrix.correlation, weighting_matrix; doDykstra = doDykstra, stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    return CovarianceMatrix(regularised_correl, covariance_matrix.volatility, covariance_matrix.labels)
end
function nearest_correlation_matrix(mat::Hermitian, ts::SortedDataFrame; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(mat).(I(size(mat)[1]))),
                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    return nearest_correlation_matrix(mat; weighting_matrix = weighting_matrix,
                                 doDykstra = doDykstra, stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
end
function nearest_correlation_matrix(mat::Hermitian; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(mat).(I(size(mat)[1]))),
                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    regularised_correl, counter, convergence = nearest_correlation_matrix(mat, weighting_matrix; doDykstra = doDykstra, stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    return Hermitian(regularised_correl)
end


"""
This function maps a Hermitian matrix to the nearest psd matrix. This uses the project_to_S
method in Higham (2001; Theorem 3.2). No special weighting is applied in this case.
Advanced users can use the `project_to_S` directly if they want to use weights in
order to decide what the `closest` pds matrix.

    nearest_psd_matrix(mat::Hermitian)
    nearest_psd_matrix(covariance_matrix::CovarianceMatrix; apply_to_covariance::Bool = true)
    nearest_psd_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; apply_to_covariance::Bool = true)
If a `Hermitian` is input then a `Hermitian` will be returned. If a `CovarianceMatrix` is
input then a `CovarianceMatrix` will be returned.


### References
Higham NJ (2002). "Computing the nearest correlation matrix - a problem from finance." IMA Journal of Numerical Analysis, 22, 329–343. doi:10.1002/nla.258.
"""
function nearest_psd_matrix(mat::Hermitian)
    W = Diagonal(Float64.(I(size(mat)[1])))
    W_root = sqrt_psd(W)
    return project_to_S(mat, W_root)
end
function nearest_psd_matrix(covariance_matrix::CovarianceMatrix; apply_to_covariance::Bool = true)
    if apply_to_covariance
        regularised_covariance = nearest_psd_matrix(covariance(covariance_matrix,1))
        corr, vols = cov2cor_and_vol(regularised_covariance, 1)
        return CovarianceMatrix(corr, vols, covariance_matrix.labels)
    else
        return CovarianceMatrix(Hermitian(nearest_psd_matrix(covariance_matrix.correlation)), covariance_matrix.volatility, covariance_matrix.labels)
    end
end
function nearest_psd_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; apply_to_covariance::Bool = true)
    return nearest_psd_matrix(covariance_matrix; apply_to_covariance = apply_to_covariance)
end
