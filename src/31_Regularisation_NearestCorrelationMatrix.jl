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

### Takes
* A - An array that you want to preject to U space.
* invW - An inverse weighting matrix.
### Returns
* A matrix that is close the input one but is unit diagonal.
# References
Higham, N. J. 2001. Bottom of page 335.
# Example
A = Hermitian([1.02 0.5 0.9; 0.5 0.98 -0.9; 0.9 -0.9 1.01])
W_root = Hermitian([1 0.5 0.5; 0.5 1.0 0.5; 0.5 0.5 1.0])
invW = inv(W_root * W_root)
project_to_U(A, invW)
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

### Takes
* A - An array that you want to preject to S space.
* W_root - The principal square root of W.
* W_inv_sqrt - The principal square root of the inverse of W.
### Returns
* A matrix that psd
# References
Higham, N. J. 2001. Theorem 3.2
# Example
A = Hermitian([1.02 0.5 0.9; 0.5 0.98 -0.9; 0.9 -0.9 1.01])
W_root = Hermitian([1 0.5 0.5; 0.5 1.0 0.5; 0.5 0.5 1.0])
invW = inv(W_root * W_root)
project_to_U(A, invW)
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

### Takes
* Y - An array that you want to project to a correlation matrix.
* Dykstra - The Dykstra corection to use. Must be the same dimensions as Y.
* W_root - The principal square root of W.
* W_inv - The invese of W.
* W_inv_sqrt - The principal square root of the inverse of W.
### Returns
* An updated matrix that is closer to be a correlation matrix.
* An updated Dykstra correction for the next iterate.
# References
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

### Takes
* A - An array that you want to project to a correlation matrix.
* W - A weighting matrix representing what correlations are thought to be more credible. There are performance improvements if this is diagonal but can be psd Hermitian as well.
* doDykstra - A bool. If False no Dykstra correction is done so the iterations have a bias but might be faster.
* stop_at_first_psd - Should the iterates stop as soon as they reach a psd matrix.
* max_iterates - The maximum number of iterates.
### Returns
* An updated matrix
* An Int describing the number of iterates done.
* A convergence message (a symbol)
# References
Higham, N. J. 2001.
"""
function nearest_correlation_matrix(A::Union{Diagonal,Hermitian}, W::Union{Diagonal,Hermitian} = Diagonal(Float64.(I(size(A)[1]))); doDykstra::Bool = true, stop_at_first_psd::Bool = true, max_iterates::Integer = 100)
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
        if stop_at_first_psd
             spd = (minimum(eigen(Y).values) > 0)
             if spd return Y, counter, :already_spd end
        end
        counter += 1
    end
    return (updated_matrix = Y, counter = counter, convergence = :reached_maxiter)
end

"""
Do one iterate mapping the matrix Y to the S space and then the U space. Returning the updated matrix and the next iterate's Dykstra correction.

### Takes
* covariance_matrix - A CovarianceMatrix
* ts - A SortedDataFrame
* weighting_matrix - A weighting matrix representing what correlations are thought to be more credible. There are performance improvements if this is diagonal but can be psd Hermitian as well.
* doDykstra - A bool. If False no Dykstra correction is done so the iterations have a bias but might be faster.
* stop_at_first_psd - Should the iterates stop as soon as they reach a psd matrix.
* max_iterates - The maximum number of iterates.
### Returns
* A CovarianceMatrix with a valid correlation matrix.
# References
Higham, N. J. 2001.
"""
function nearest_correlation_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; weighting_matrix = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                             doDykstra::Bool = true, stop_at_first_psd::Bool = true, max_iterates::Integer = 100)
    regularised_correl, counter, convergence = nearest_correlation_matrix(covariance_matrix.correlation, weighting_matrix; doDykstra = doDykstra, stop_at_first_psd = stop_at_first_psd, max_iterates = max_iterates)
    return CovarianceMatrix(regularised_correl, covariance_matrix.volatility, covariance_matrix.labels)
end

function nearest_psd_matrix(mat::Hermitian, ts::SortedDataFrame)
    W = Diagonal(Float64.(I(size(mat)[1])))
    W_root = sqrt_psd(W)
    return project_to_S(mat, W_root)
end
