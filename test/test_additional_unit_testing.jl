using Test

@testset "Additional unit testing" begin

    using LinearAlgebra
    using Distributions
    using StableRNGs
    using HighFrequencyCovariance


    # Testing the function for regularising a Hermitian.
    A = Hermitian([1.02 0.5 0.9; 0.5 0.98 0.9; 0.9 0.9 1.01])
    B = eigenvalue_clean(A, 0.0000001)
    @test isa(B, Hermitian)


    # Testing the case when there are nans in an estimated matrix.
    brownian_corr_matrix = Hermitian([
        1.0 0.75 0.5 0.0
        0.0 1.0 0.5 0.25
        0.0 0.0 1.0 0.25
        0.0 0.0 0.0 1.0
    ])
    assets = [:BARC, :HSBC, :VODL, :RYAL]
    rng = StableRNG(1)
    rng2 = StableRNG(2)
    ts1, true_covar, micro_noise, update_rates = generate_random_path(
        4,
        5000;
        brownian_corr_matrix = brownian_corr_matrix,
        assets = assets,
        vols = [0.02, 0.03, 0.04, 0.05],
        rng = deepcopy(rng),
        rng_timing = deepcopy(rng2),
    )
    # For the purpose of testing we assume that we got the following matrix from some estimation on this ts1
    estimated = Matrix(copy(true_covar.correlation))
    estimated[1,2] = NaN
    estimated = Hermitian(estimated)
    estimated2 = eigenvalue_clean(estimated, ts1)
    # As this cannot be regularised the input should be returned unchanged.
    @test isnan(estimated2[1,2])
end
