using Test
using LinearAlgebra

# Copypasting this here to avoid exporting it
function sqrt_psd(A::Hermitian)
    eigenvalues, eigenvectors = eigen(A)
    return Hermitian(construct_matrix_from_eigen(sqrt.(eigenvalues), eigenvectors))
end
sqrt_psd(A::Diagonal) = sqrt(A)

const TOLTOL = 10_000 * eps()

@testset "Test Higham regularisation - slightly non psd" begin

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
    @test all(abs.(diag(U_ed) .- 1.0) .< TOLTOL)
    @test isa(U_ed, Hermitian)
    S_ed = project_to_S(A, W_root)
    @test all(eigen(S_ed).values .> -TOLTOL)


    # hermitian weighting.
    W_root = Hermitian(rand(twister, IW_dist(3)))
    invW = inv(W_root^2)
    U_ed = project_to_U(A, invW)
    @test all(abs.(diag(U_ed) .- 1.0) .< TOLTOL)
    @test isa(U_ed, Hermitian)
    S_ed = project_to_S(A, W_root)
    @test all(eigen(S_ed).values .> -TOLTOL)
end


@testset "Test Higham regularisation - really non psd" begin

    using LinearAlgebra
    using Distributions: InverseWishart
    using Random
    using HighFrequencyCovariance
    twister = MersenneTwister(10)
    IW_dist(n) = InverseWishart(n, Matrix(Float64.(I(n))))

    # Really non psd non diagonal version.
    A = Hermitian([1.02 0.5 0.9; 0.5 0.98 -0.9; 0.9 -0.9 1.01])
    # Diagonal weighting.
    W_root = Diagonal(rand(twister, size(A)[1]))
    invW = inv(W_root^2)
    U_ed = project_to_U(A, invW)
    @test all(abs.(diag(U_ed) .- 1.0) .< TOLTOL)
    @test isa(U_ed, Hermitian)
    S_ed = project_to_S(A, W_root)
    @test all(eigen(S_ed).values .> -TOLTOL)

    # hermitian weighting.
    W_root = Hermitian(rand(twister, IW_dist(3)))
    invW = inv(W_root^2)
    U_ed = project_to_U(A, invW)
    @test all(abs.(diag(U_ed) .- 1.0) .< TOLTOL)
    @test isa(U_ed, Hermitian)
    S_ed = project_to_S(A, W_root)
    @test all(eigen(S_ed).values .> -TOLTOL)

    # Diagonal A
    A = Diagonal([1.02 0 0; 0 0.98 0; 0 0 1.05])
    # Diagonal weighting.
    W_root = Diagonal(rand(twister, size(A)[1]))
    invW = inv(W_root^2)
    U_ed = project_to_U(A, invW)
    @test all(abs.(diag(U_ed) .- 1.0) .< TOLTOL)
    @test isa(U_ed, Diagonal)
    S_ed = project_to_S(A, W_root)
    @test all(eigen(S_ed).values .> -TOLTOL)

    # hermitian weighting.
    W_root = Hermitian(rand(twister, IW_dist(3)))
    invW = inv(W_root^2)
    U_ed = project_to_U(A, invW)
    @test all(abs.(diag(U_ed) .- 1.0) .< TOLTOL)
    @test isa(U_ed, Hermitian)
    S_ed = project_to_S(A, W_root)
    @test all(eigen(S_ed).values .> -TOLTOL)

end

@testset "Test Higham regularisation - internals" begin

    using LinearAlgebra
    using Distributions: InverseWishart
    using Random
    using HighFrequencyCovariance
    twister = MersenneTwister(10)
    IW_dist(n) = InverseWishart(n, Matrix(Float64.(I(n))))

    # A identify matrix case.
    N = 4
    A = Hermitian(I(N))
    W_root = Hermitian(rand(twister, IW_dist(N)))
    Dykstra = Hermitian(zeros(N, N))
    W_inv = inv(W_root^2)
    W_inv_sqrt = sqrt_psd(W_inv)
    Y = A

    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- I(N)) .< TOLTOL)
    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- I(N)) .< TOLTOL)

    # If A were input as diagonal
    A = Diagonal(I(N))
    Dykstra = Diagonal(zeros(N, N))
    Y = A
    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- I(N)) .< TOLTOL)
    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- I(N)) .< TOLTOL)

    # If A and E_root were diagonal
    A = Diagonal(Float64.(I(N)))
    W_root = Diagonal(rand(twister, N))
    Dykstra = Diagonal(zeros(N, N))
    W_inv = inv(W_root^2)
    W_inv_sqrt = sqrt_psd(W_inv)
    Y = A
    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- I(N)) .< TOLTOL)
    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- I(N)) .< TOLTOL)
end

@testset "Test Higham regularisation - valid psd case" begin
    using LinearAlgebra
    using Distributions: InverseWishart
    using Random
    using HighFrequencyCovariance
    twister = MersenneTwister(10)
    IW_dist(n) = InverseWishart(n, Matrix(Float64.(I(n))))
        
    # A valid correlation matrix case
    N = 3
    A, _ = cov_to_cor(Hermitian(rand(twister, IW_dist(N))))
    W_root = Hermitian(rand(twister, IW_dist(N)))
    Dykstra = Hermitian(zeros(N, N))
    W_inv = inv(W_root^2)
    W_inv_sqrt = sqrt_psd(W_inv)
    Y = A

    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- A) .< 100 * eps())
    Y, Dykstra = iterate_higham(Y, Dykstra, W_root, W_inv, W_inv_sqrt)
    @test all(abs.(Y .- A) .< 100 * eps())
end

@testset "Test Higham regularisation - another non psd case" begin
    using LinearAlgebra
    using Distributions: InverseWishart
    using Random
    using HighFrequencyCovariance
    twister = MersenneTwister(10)
    IW_dist(n) = InverseWishart(n, Matrix(Float64.(I(n))))

    # With diagonal weighting.
    N = 3
    A = Hermitian([1.02 0.5 0.9; 0.5 0.98 0.9; 0.9 0.9 1.01])
    W = Diagonal(rand(twister, N))
    updated_matrix, counter, convergence = nearest_correlation_matrix(
        A,
        W;
        doDykstra = true,
        stop_at_first_correlation_matrix = true,
        max_iterates = 1000,
    )
    updated_matrix2, counter2, convergence2 = nearest_correlation_matrix(
        A,
        W;
        doDykstra = true,
        stop_at_first_correlation_matrix = false,
        max_iterates = 10000,
    )
    updated_matrix3, counter3, convergence3 = nearest_correlation_matrix(
        A,
        W;
        doDykstra = false,
        stop_at_first_correlation_matrix = true,
        max_iterates = 10000,
    )
    @test all(abs.(updated_matrix .- updated_matrix2) .< TOLTOL) # These should be the same as once it is psd the matrix stops updating.
    @test any(abs.(updated_matrix .- updated_matrix3) .> 0.000000001) # These should be different as without Dykstra one is biased
    # Testing convergences.
    @test minimum(eigen(updated_matrix2).values) > -TOLTOL
    @test minimum(eigen(updated_matrix).values) > -TOLTOL
    @test minimum(eigen(updated_matrix3).values) > -TOLTOL
end
