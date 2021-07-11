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
Do one iterate mapping the matrix Y to the S space and then the U space. Returning the updated matrix and the next iterate's Dykstra correction.
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
Do one iterate mapping the matrix Y to the S space and then the U space. Returning the updated matrix and the next iterate's Dykstra correction.
Higham, N. J. 2001.
"""
function nearest_correlation_matrix(A::Union{Diagonal,Hermitian}, W::Union{Diagonal,Hermitian} = Diagonal(Float64.(I(size(A)[1]))); doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    @assert all(size(A) .== size(W))
    W_root = sqrt_psd(W)
    W_inv = inv(W)
    W_inv_sqrt = sqrt_psd(W_inv)

    N = size(A)[1]
    Y = A
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

"""
Do one iterate mapping the matrix Y to the S space and then the U space. Returning the updated matrix and the next iterate's Dykstra correction.
Higham, N. J. 2001.
"""
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


function nearest_correlation_matrix(mat::Hermitian, ts::SortedDataFrame, mat_labels = missing; weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    return nearest_correlation_matrix(mat, mat_labels; weighting_matrix = weighting_matrix,
                                 doDykstra = doDykstra, stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
end
function nearest_correlation_matrix(mat::Hermitian, mat_labels::Vector = missing; weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
    regularised_correl, counter, convergence = nearest_correlation_matrix(mat, weighting_matrix; doDykstra = doDykstra, stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    return Hermitian(regularised_correl)
end


"""
Map a Hermitian matrix to the nearest psd matrix.
"""
function nearest_psd_matrix(mat::Hermitian, ts::SortedDataFrame, mat_labels = missing)
    W = Diagonal(Float64.(I(size(mat)[1])))
    W_root = sqrt_psd(W)
    return project_to_S(mat, W_root)
end
function nearest_psd_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; apply_to_covariance::Bool = true)
    if apply_to_covariance
        regularised_covariance = nearest_psd_matrix(covariance(covariance_matrix,1), ts)
        corr, vols = cov2cor_and_vol(regularised_covariance, 1)
        return CovarianceMatrix(corr, vols, covariance_matrix.labels)
    else
        return CovarianceMatrix(Hermitian(nearest_psd_matrix(covariance_matrix.correlation, ts)), covariance_matrix.volatility, covariance_matrix.labels)
    end
end
