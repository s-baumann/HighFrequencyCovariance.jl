const NUMERICAL_TOL = 10 * eps()

"""
    set_diagonal_to_one(A::Diagonal)

    set_diagonal_to_one(A::Hermitian)

This returns the matrix square root of an input matrix.
"""
function set_diagonal_to_one!(A::Diagonal)
    A[diagind(A)] .= 1
end
function set_diagonal_to_one!(A::Hermitian)
    A[diagind(A)] .= 1
end

"""
    sqrt_psd(A::Hermitian)

    sqrt_psd(A::Diagonal)

This returns the matrix square root of an input matrix.
"""
function sqrt_psd(A::Hermitian)
    eigenvalues, eigenvectors = eigen(A)
    return Hermitian(construct_matrix_from_eigen(sqrt.(eigenvalues), eigenvectors))
end
sqrt_psd(A::Diagonal) = sqrt(A)

"""
    project_to_U(A::Union{Diagonal,Hermitian}, invW::Hermitian)

    project_to_U(A::Union{Diagonal,Hermitian}, invW::Diagonal)

This maps the Hermitian/Hermitian matrix `A` to the nearest matrix in the U space (the space of all unit diagonal matrices as defined by Higham 2001). The inverse weight matrix `invW` determines how much to adjust
each element to get it to be unit diagonal. In other words it is used to determine what is the nearest correlation matrix. The weight matrix must be Hermitian positive definite.
We use the W-norm (as defined by Higham 2001).
### Inputs
* `A` - The matrix you want to project to the U space
* `invW` - The inverse weighting matrix.
### Outputs
* A `Diagonal` or a `Hermitian`.

### References
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
    set_diagonal_to_one!(A)
    return A
end

"""
    project_to_S(
        A::Hermitian,
        W_root::Union{Hermitian,Diagonal};
        W_inv_sqrt::Union{Hermitian,Diagonal} = sqrt_psd(inv(W_root^2)),
    )

    project_to_S(
        A::Diagonal,
        W_root::Union{Hermitian,Diagonal};
        W_inv_sqrt::Union{Hermitian,Diagonal,Missing} = missing,
    )

This maps a matrix to the nearest psd matrix. `W_root` should be the principal square root of a psd Hermitian weighting matrix, `W`.
`W_inv_sqrt` should be the corresponding square root of the inverse of `W`.
`nearest_psd_matrix` is a simpler interface for this function however it does not allow weighting matrices to be specified.
### Inputs
* `A` - The matrix you want to project to the S space. This can be a `Diagonal` or a `Hermitian`. Note that if you input a `Diagonal` matrix then it is already in the S space and so it will be returned without any calculation.
* `W_root` - The inverse weighting matrix.
* `W_inv_sqrt` - The root of `W_root`. This is calculated if you don't have it but it can save some calculation effort if you already have it.
### Outputs
* A `Hermitian`.

### References
Higham, N. J. 2001. Theorem 3.2
"""
function project_to_S(
    A::Hermitian,
    W_root::Union{Hermitian,Diagonal};
    W_inv_sqrt::Union{Hermitian,Diagonal} = sqrt_psd(inv(W_root^2)),
)
    Wroot_A_WRoot = Hermitian(W_root * A * W_root)
    eigenvalues, eigenvectors = eigen(Wroot_A_WRoot)
    if all(eigenvalues .> 0)
        return A
    end
    positive_eigenvalues = map(x -> max(0, x), eigenvalues)
    Wroot_A_WRoot_positiveEigen =
        W_inv_sqrt *
        construct_matrix_from_eigen(positive_eigenvalues, eigenvectors) *
        W_inv_sqrt
    return Hermitian(Wroot_A_WRoot_positiveEigen)
end
project_to_S(
    A::Diagonal,
    W_root::Union{Hermitian,Diagonal};
    W_inv_sqrt::Union{Hermitian,Diagonal,Missing} = missing,
) = A


"""
    iterate_higham(
       Y::Union{Hermitian,Diagonal},
       Dykstra::Union{Hermitian,Diagonal},
       W_root::Union{Hermitian,Diagonal},
       W_inv::Union{Hermitian,Diagonal},
       W_inv_sqrt::Union{Hermitian,Diagonal},
    )

Do one iterate mapping the input matrix to the S space (of psd matrices) and then to the U space (unit diagonal and all other entries below 1 in absolute value).
Returns the updated matrix and the next iterate's Dykstra correction.
### Inputs
* `Y` - The matrix you want to project to the iterate towards the space of valid correlation matrices.
* `Dykstra` - The Dykstra correction matrix.
* `W_root` - The root of `W`.
* `W_inv` - The inverse of `W`.
* `W_inv_sqrt` - The root of the inverse of `W`.
### Outputs
* A `Hermitian`.
* An updated Dykstra correction matrix.

### References
Higham, N. J. 2001. Algorithm 3.3
"""
function iterate_higham(
    Y::Union{Hermitian,Diagonal},
    Dykstra::Union{Hermitian,Diagonal},
    W_root::Union{Hermitian,Diagonal},
    W_inv::Union{Hermitian,Diagonal},
    W_inv_sqrt::Union{Hermitian,Diagonal},
)
    R = Y - Dykstra
    X = project_to_S(R, W_root; W_inv_sqrt = W_inv_sqrt)
    new_Dykstra = X - R
    new_Y = project_to_U(X, W_inv)
    return new_Y, new_Dykstra
end

"""
    nearest_correlation_matrix(
        mat::AbstractMatrix,
        weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(Float64.(I(size(mat)[1])));
        doDykstra::Bool = true,
        stop_at_first_correlation_matrix::Bool = true,
        max_iterates::Integer = 1000,
    )

Maps a matrix to the nearest valid correlation matrix (pdf matrix with unit diagonal and all other entries below 1 in absolute value).
### Inputs
* `mat` - A matrix you want to regularise.
* `ts` - The tick data.
* `weighting_matrix` - The weighting matrix used to weight what the **nearest** valid correlation matrix is.
* `doDykstra` - Should Dykstra correction be done.
* `stop_at_first_correlation_matrix` - Should we keep iterating until we have done all iterates or stop at the first valid correlation matrix.
* `max_iterates` - The maximum number of iterates to do towards a valid correlation matrix.
### Returns
* A `Matrix`
* An integer saying how many iterates were done
* A Symbol with a convergence message.

    nearest_correlation_matrix(
        covariance_matrix::CovarianceMatrix,
        ts::SortedDataFrame;
        weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
        doDykstra::Bool = true,
        stop_at_first_correlation_matrix::Bool = true,
        max_iterates::Integer = 1000,
    )

Maps a matrix to the nearest valid correlation matrix (pdf matrix with unit diagonal and all other entries below 1 in absolute value).
### Inputs
* `covariance_matrix` - The matrix you want to regularise.
* `ts` - The tick data.
* `weighting_matrix` - The weighting matrix used to weight what the **nearest** valid correlation matrix is.
* `doDykstra` - Should Dykstra correction be done.
* `stop_at_first_correlation_matrix` - Should we keep iterating until we have done all iterates or stop at the first valid correlation matrix.
* `max_iterates` - The maximum number of iterates to do towards a valid correlation matrix.
### Returns
* A `CovarianceMatrix`

    nearest_correlation_matrix(
        covariance_matrix::CovarianceMatrix;
        weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
        doDykstra::Bool = true,
        stop_at_first_correlation_matrix::Bool = true,
        max_iterates::Integer = 1000,
    )

Maps a matrix to the nearest valid correlation matrix (pdf matrix with unit diagonal and all other entries below 1 in absolute value).
### Inputs
* `covariance_matrix` - The matrix you want to regularise.
* `weighting_matrix` - The weighting matrix used to weight what the **nearest** valid correlation matrix is.
* `doDykstra` - Should Dykstra correction be done.
* `stop_at_first_correlation_matrix` - Should we keep iterating until we have done all iterates or stop at the first valid correlation matrix.
* `max_iterates` - The maximum number of iterates to do towards a valid correlation matrix.
### Returns
* A `CovarianceMatrix`

    nearest_correlation_matrix(
        mat::Hermitian,
        ts::SortedDataFrame;
        weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(mat).(I(size(mat)[1]))),
        doDykstra::Bool = true,
        stop_at_first_correlation_matrix::Bool = true,
        max_iterates::Integer = 1000,
    )

Maps a matrix to the nearest valid correlation matrix (pdf matrix with unit diagonal and all other entries below 1 in absolute value).
### Inputs
* `mat` - The matrix you want to regularise.
* `weighting_matrix` - The weighting matrix used to weight what the **nearest** valid correlation matrix is.
* `doDykstra` - Should Dykstra correction be done.
* `stop_at_first_correlation_matrix` - Should we keep iterating until we have done all iterates or stop at the first valid correlation matrix.
* `max_iterates` - The maximum number of iterates to do towards a valid correlation matrix.
### Returns
* A `Hermitian`

    nearest_correlation_matrix(
       mat::Hermitian;
       weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(mat).(I(size(mat)[1]))),
       doDykstra::Bool = true,
       stop_at_first_correlation_matrix::Bool = true,
       max_iterates::Integer = 1000,
    )

Maps a matrix to the nearest valid correlation matrix (pdf matrix with unit diagonal and all other entries below 1 in absolute value).
### Inputs
* `covariance_matrix` - The matrix you want to regularise.
* `ts` - The tick data.
* `weighting_matrix` - The weighting matrix used to weight what the **nearest** valid correlation matrix is.
* `doDykstra` - Should Dykstra correction be done.
* `stop_at_first_correlation_matrix` - Should we keep iterating until we have done all iterates or stop at the first valid correlation matrix.
* `max_iterates` - The maximum number of iterates to do towards a valid correlation matrix.
### Returns
* A `Hermitian`

### References
Higham NJ (2002). "Computing the nearest correlation matrix - a problem from finance." IMA Journal of Numerical Analysis, 22, 329–343. doi:10.1002/nla.258.
"""
function nearest_correlation_matrix(
    mat::AbstractMatrix,
    weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(Float64.(I(size(mat)[1])));
    doDykstra::Bool = true,
    stop_at_first_correlation_matrix::Bool = true,
    max_iterates::Integer = 1000,
)
    @assert all(size(mat) .== size(weighting_matrix))
    W_root = sqrt_psd(weighting_matrix)
    W_inv = inv(weighting_matrix)
    W_inv_sqrt = sqrt_psd(W_inv)

    N = size(mat)[1]
    Y = mat
    ZeroDykstra = Hermitian(zeros(N, N))
    Dykstra = ZeroDykstra
    counter = 1
    while counter < max_iterates
        Y, Dykstra = doDykstra ? iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt) :
            iterate_higham(Y, ZeroDykstra, W_root, W_inv, W_inv_sqrt)
        if stop_at_first_correlation_matrix
            spd = valid_correlation_matrix(Y)
            if spd
                return Y, counter, :already_valid_correlation_matrix
            end
        end
        counter += 1
    end
    return (updated_matrix = Y, counter = counter, convergence = :reached_maxiter)
end
function nearest_correlation_matrix(
    covariance_matrix::CovarianceMatrix,
    ts::SortedDataFrame;
    weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
    doDykstra::Bool = true,
    stop_at_first_correlation_matrix::Bool = true,
    max_iterates::Integer = 1000,
)
    return nearest_correlation_matrix(
        covariance_matrix;
        weighting_matrix = weighting_matrix,
        doDykstra = doDykstra,
        stop_at_first_correlation_matrix = stop_at_first_correlation_matrix,
        max_iterates = max_iterates,
    )
end
function nearest_correlation_matrix(
    covariance_matrix::CovarianceMatrix;
    weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
    doDykstra::Bool = true,
    stop_at_first_correlation_matrix::Bool = true,
    max_iterates::Integer = 1000,
)
    regularised_correl, counter, convergence = nearest_correlation_matrix(
        covariance_matrix.correlation,
        weighting_matrix;
        doDykstra = doDykstra,
        stop_at_first_correlation_matrix = stop_at_first_correlation_matrix,
        max_iterates = max_iterates,
    )
    return CovarianceMatrix(
        regularised_correl,
        covariance_matrix.volatility,
        covariance_matrix.labels,
        covariance_matrix.time_period_per_unit,
    )
end
function nearest_correlation_matrix(
    mat::Hermitian,
    ts::SortedDataFrame;
    weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(mat).(I(size(mat)[1]))),
    doDykstra::Bool = true,
    stop_at_first_correlation_matrix::Bool = true,
    max_iterates::Integer = 1000,
)
    return nearest_correlation_matrix(
        mat;
        weighting_matrix = weighting_matrix,
        doDykstra = doDykstra,
        stop_at_first_correlation_matrix = stop_at_first_correlation_matrix,
        max_iterates = max_iterates,
    )
end
function nearest_correlation_matrix(
    mat::Hermitian;
    weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(mat).(I(size(mat)[1]))),
    doDykstra::Bool = true,
    stop_at_first_correlation_matrix::Bool = true,
    max_iterates::Integer = 1000,
)
    regularised_correl, counter, convergence = nearest_correlation_matrix(
        mat,
        weighting_matrix;
        doDykstra = doDykstra,
        stop_at_first_correlation_matrix = stop_at_first_correlation_matrix,
        max_iterates = max_iterates,
    )
    return Hermitian(regularised_correl)
end


"""
    nearest_psd_matrix(mat::Hermitian)

This function maps a Hermitian matrix to the nearest psd matrix. This uses the `project_to_S`
method in Higham (2001; Theorem 3.2). No special weighting is applied in this case.
Advanced users can use the `project_to_S` directly if they want to use weights in
order to decide what the `closest` pds matrix.
### Inputs
* `mat` - The matrix you want to map to a psd matrix
### Results
* A `Hermitian`

    nearest_psd_matrix(
        covariance_matrix::CovarianceMatrix;
        apply_to_covariance::Bool = true,
    )

This function maps a Hermitian matrix to the nearest psd matrix. This uses the `project_to_S`
method in Higham (2001; Theorem 3.2). No special weighting is applied in this case.
Advanced users can use the `project_to_S` directly if they want to use weights in
order to decide what the `closest` pds matrix.
### Inputs
* `covariance_matrix` - The matrix you want to map to a psd matrix
* `apply_to_covariance` - Should regularisation be applied to the correlation or covariance matrix.
### Results
* A `CovarianceMatrix`

    nearest_psd_matrix(
        covariance_matrix::CovarianceMatrix,
        ts::SortedDataFrame;
        apply_to_covariance::Bool = true,
    )

This function maps a Hermitian matrix to the nearest psd matrix. This uses the `project_to_S`
method in Higham (2001; Theorem 3.2). No special weighting is applied in this case.
Advanced users can use the `project_to_S` directly if they want to use weights in
order to decide what the `closest` pds matrix.
### Inputs
* `covariance_matrix` - The matrix you want to map to a psd matrix
* `ts` - The Tick data
* `apply_to_covariance` - Should regularisation be applied to the correlation or covariance matrix.
### Results
* A `CovarianceMatrix`

### References
Higham NJ (2002). "Computing the nearest correlation matrix - a problem from finance." IMA Journal of Numerical Analysis, 22, 329–343. doi:10.1002/nla.258.
"""
function nearest_psd_matrix(mat::Hermitian)
    W = Diagonal(Float64.(I(size(mat)[1])))
    W_root = sqrt_psd(W)
    projected = project_to_S(mat, W_root)
    # This is to avoid really slight negative eigenvalues like -1E-17 that happen due to computational rounding.
    # At this is s a numerical issue we adjust by a multiple of the machine epsilon. In tests 100 times was
    # a small miltiple of epsilon that prevented the issue.
    if !is_psd_matrix(projected)
        projected = identity_regularisation(projected, 100 * eps())
    end
    return projected
end
function nearest_psd_matrix(
    covariance_matrix::CovarianceMatrix;
    apply_to_covariance::Bool = true,
)
    if apply_to_covariance
        regularised_covariance = nearest_psd_matrix(covariance(covariance_matrix))
        corr, vols = cov_to_cor_and_vol(regularised_covariance, 1)
        return CovarianceMatrix(
            corr,
            vols,
            covariance_matrix.labels,
            covariance_matrix.time_period_per_unit,
        )
    else
        return CovarianceMatrix(
            Hermitian(nearest_psd_matrix(covariance_matrix.correlation)),
            covariance_matrix.volatility,
            covariance_matrix.labels,
            covariance_matrix.time_period_per_unit,
        )
    end
end
function nearest_psd_matrix(
    covariance_matrix::CovarianceMatrix,
    ts::SortedDataFrame;
    apply_to_covariance::Bool = true,
)
    return nearest_psd_matrix(covariance_matrix; apply_to_covariance = apply_to_covariance)
end
