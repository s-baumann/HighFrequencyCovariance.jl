using LinearAlgebra
using Distributions: InverseWishart
using Random
using HighFrequencyCovariance

twister = MersenneTwister(10)
IW_dist(n) = InverseWishart(n, Matrix(Float64.(I(n))))

# slightly non psd non diagonal version.
A = Hermitian([1.02 0.5 0.9; 0.5 0.98 0.9; 0.9 0.9 1.01])
# Diagonal weighting.
W_root = Diagonal(rand(twister, size(A)[1]))
invW = inv(W_root^2)
U_ed = project_to_U(A, invW)
all( abs.(diag(U_ed) .- 1.0) .< 100*eps())
isa(U_ed, Hermitian)
S_ed = project_to_S(A, W_root)
all(eigen(S_ed).values .> -100*eps())


# hermitian weighting.
W_root = Hermitian(rand(twister, IW_dist(3)  ))
invW = inv(W_root^2)
U_ed = project_to_U(A, invW)
all( abs.(diag(U_ed) .- 1.0) .< 100*eps())
isa(U_ed, Hermitian)
S_ed = project_to_S(A, W_root)
all(eigen(S_ed).values .> -100*eps())


# Really non psd non diagonal version.
A = Hermitian([1.02 0.5 0.9; 0.5 0.98 -0.9; 0.9 -0.9 1.01])
# Diagonal weighting.
W_root = Diagonal(rand(twister, size(A)[1]))
invW = inv(W_root^2)
U_ed = project_to_U(A, invW)
all( abs.(diag(U_ed) .- 1.0) .< 100*eps())
isa(U_ed, Hermitian)
S_ed = project_to_S(A, W_root)
all(eigen(S_ed).values .> -100*eps())

# hermitian weighting.
W_root = Hermitian(rand(twister, IW_dist(3)  ))
invW = inv(W_root^2)
U_ed = project_to_U(A, invW)
all( abs.(diag(U_ed) .- 1.0) .< 100*eps())
isa(U_ed, Hermitian)
S_ed = project_to_S(A, W_root)
all(eigen(S_ed).values .> -100000*eps())

# Diagonal A
A = Diagonal([1.02 0 0; 0 0.98 0; 0 0 1.05])
# Diagonal weighting.
W_root = Diagonal(rand(twister, size(A)[1]))
invW = inv(W_root^2)
U_ed = project_to_U(A, invW)
all( abs.(diag(U_ed) .- 1.0) .< 100*eps())
isa(U_ed, Diagonal)
S_ed = project_to_S(A, W_root)
all(eigen(S_ed).values .> -100*eps())

# hermitian weighting.
W_root = Hermitian(rand(twister, IW_dist(3)  ))
invW = inv(W_root^2)
U_ed = project_to_U(A, invW)
all( abs.(diag(U_ed) .- 1.0) .< 100*eps())
isa(U_ed, Hermitian)
S_ed = project_to_S(A, W_root)
all(eigen(S_ed).values .> -100000*eps())






# Copypasting this here to avoid exporting it
function sqrt_psd(A::Hermitian)
    eigenvalues, eigenvectors = eigen(A)
    return Hermitian(construct_matrix_from_eigen(sqrt.(eigenvalues), eigenvectors))
end
sqrt_psd(A::Diagonal) = sqrt(A)




# A identify matrix case.
N = 4
A = Hermitian(I(N))
W_root = Hermitian(rand(twister, IW_dist(N)  ))
Dykstra = Hermitian(zeros(N,N))
W_inv = inv(W_root^2)
W_inv_sqrt = sqrt_psd(W_inv)
Y = A

Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- I(N)) .< 100*eps())
Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- I(N)) .< 100*eps())

# If A were input as diagonal
A = Diagonal(I(N))
Dykstra = Diagonal(zeros(N,N))
Y = A
Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- I(N)) .< 100*eps())
Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- I(N)) .< 100*eps())

# If A and E_root were diagonal
A = Diagonal(Float64.(I(N)))
W_root = Diagonal(rand(twister, N))
Dykstra = Diagonal(zeros(N,N))
W_inv = inv(W_root^2)
W_inv_sqrt = sqrt_psd(W_inv)
Y = A
Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- I(N)) .< 100*eps())
Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- I(N)) .< 100*eps())



# A valid correlation matrix case
N = 3
A, _ = cov2cor(Hermitian(rand(twister, IW_dist(N)  )))
W_root = Hermitian(rand(twister, IW_dist(N)  ))
Dykstra = Hermitian(zeros(N,N))
W_inv = inv(W_root^2)
W_inv_sqrt = sqrt_psd(W_inv)
Y = A

Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- A) .< 100*eps())
Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
all(abs.(Y .- A) .< 100*eps())


# A Slightly non psd case
N = 3
A = Hermitian([1.02 0.5 0.9; 0.5 0.98 0.9; 0.9 0.9 1.01])
W = Hermitian(abs.(cov2cor(Hermitian(rand(twister, IW_dist(N)  )))[1]))
updated_matrix, counter, convergence = nearest_correlation_matrix(A, W; doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
updated_matrix2, counter2, convergence2 = nearest_correlation_matrix(A, W; doDykstra = true, stop_at_first_correlation_matrix = false, max_iterates = 10000)
updated_matrix3, counter3, convergence3 = nearest_correlation_matrix(A, W; doDykstra = false, stop_at_first_correlation_matrix = true, max_iterates = 10000)
all(abs.(updated_matrix .- updated_matrix2) .< 100*eps()) # These should be the same as once it is psd the matrix stops updating.
any(abs.(updated_matrix .- updated_matrix3) .> 0.0001) # These should be different as without Dykstra one is biased
# Testing convergences.
minimum(eigen(updated_matrix).values) > -10*eps()
minimum(eigen(updated_matrix2).values) > -10*eps()
minimum(eigen(updated_matrix3).values) > -10*eps()


# With diagonal weighting.
N = 3
A = Hermitian([1.02 0.5 0.9; 0.5 0.98 0.9; 0.9 0.9 1.01])
W = Diagonal(rand(twister,N))
updated_matrix, counter, convergence = nearest_correlation_matrix(A, W; doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
updated_matrix2, counter2, convergence2 = nearest_correlation_matrix(A, W; doDykstra = true, stop_at_first_correlation_matrix = false, max_iterates = 10000)
updated_matrix3, counter3, convergence3 = nearest_correlation_matrix(A, W; doDykstra = false, stop_at_first_correlation_matrix = true, max_iterates = 10000)
all(abs.(updated_matrix .- updated_matrix2) .< 100*eps()) # These should be the same as once it is psd the matrix stops updating.
any(abs.(updated_matrix .- updated_matrix3) .> 0.0001) # These should be different as without Dykstra one is biased
# Testing convergences.
minimum(eigen(updated_matrix).values) > -10*eps()
minimum(eigen(updated_matrix2).values) > -10*eps()
minimum(eigen(updated_matrix3).values) > -10*eps()

# A really non psd case
A = Hermitian([1.02 0.5 0.9; 0.5 0.98 -0.9; 0.9 -0.9 1.01])
W = Hermitian(abs.(cov2cor(Hermitian(rand(twister, IW_dist(N)  )))[1]))
updated_matrix, counter, convergence = nearest_correlation_matrix(A, W; doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 100000)
updated_matrix2, counter2, convergence2 = nearest_correlation_matrix(A, W; doDykstra = true, stop_at_first_correlation_matrix = false, max_iterates = 10000)
updated_matrix3, counter3, convergence3 = nearest_correlation_matrix(A, W; doDykstra = false, stop_at_first_correlation_matrix = true, max_iterates = 10000)
all(abs.(updated_matrix .- updated_matrix2) .< 1000*eps()) # These should be the same as once it is psd the matrix stops updating.
any(abs.(updated_matrix .- updated_matrix3) .> 0.0001) # These should be different as without Dykstra one is biased
# Testing convergences.
minimum(eigen(updated_matrix).values) > -10*eps()
minimum(eigen(updated_matrix2).values) > -10*eps()
