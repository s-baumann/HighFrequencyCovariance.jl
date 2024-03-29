using Test

@testset "Test Convenience Functions" begin

    using DataFrames
    using LinearAlgebra
    using Statistics: std, var, mean, cov
    using HighFrequencyCovariance
    using StableRNGs
    using Random

    brownian_corr_matrix = Hermitian([
        1.0 0.75 0.5 0.0
        0.0 1.0 0.5 0.25
        0.0 0.0 1.0 0.25
        0.0 0.0 0.0 1.0
    ])
    assets = [:BARC, :HSBC, :VODL, :RYAL]
    rng = StableRNG(1)

    ts1, true_covar, micro_noise, update_rates = generate_random_path(
        4,
        2000;
        brownian_corr_matrix = brownian_corr_matrix,
        assets = assets,
        vols = [0.02, 0.03, 0.04, 0.05],
        rng = deepcopy(rng),
    )
    ts1.df.value = exp.(ts1.df.Value)

    # Testing making a ItoSet
    ito = ItoSet(true_covar)


    iscloser(a, b) = (
        a.Correlation_error + a.Volatility_error < b.Correlation_error + b.Volatility_error
    )

    function subset_dict(dic::Dict, keys::Vector)
        new_dict = Dict()
        for k in keys
            new_dict[k] = dic[k]
        end
        return new_dict
    end

    # Testing Covariation matrix functions.
    covar_functions = [
        simple_covariance,
        bnhls_covariance,
        spectral_covariance,
        preaveraged_covariance,
        two_scales_covariance,
    ]
    function test_covariance(settings = Dict())
        success = true
        for f in covar_functions
            k = Symbol(f)
            args = reduce(union, Base.kwarg_decl.(methods(f)))
            releventsettings = intersect(args, keys(settings))
            sets =
                (length(releventsettings) > 0) ? subset_dict(settings, releventsettings) :
                Dict()
            estimate = f(ts1, assets; sets...)
            estimate_c = estimate_covariance(ts1, assets, k; sets...)
            cor_err = calculate_mean_abs_distance(estimate, estimate_c).Correlation_error
            t1 = (cor_err .< 10 * eps()) | isnan(cor_err)
            if (t1 == false)
                println(
                    "In convenience function covariance test there is a covariance error for ",
                    k,
                )
            end
            vol_err = calculate_mean_abs_distance(estimate, estimate_c).Volatility_error
            t2 = (vol_err .< 10 * eps()) | isnan(vol_err)
            if (t2 == false)
                println(
                    "In convenience function covariance test there is a volatility error for ",
                    k,
                )
            end
            success = success & (t1 & t2)
        end
        return success
    end
    @test test_covariance()
    settings1 = Dict([:kernel, :num_grids] .=> [fejer, 15])
    @test test_covariance(settings1)
    settings2 = Dict(
        [:theta, :num_grids, :only_regulise_if_not_PSD, :regularisation, :equalweight] .=> [0.5, 20, true, :identity_regularisation, true],
    )
    @test test_covariance(settings2)
    settings3 = Dict{Symbol,Any}([:numJ, :num_blocks, :block_width] .=> [Int(110), 30, 20])
    @test test_covariance(settings3)


    # Volatility
    vol_functions = [simple_volatility, two_scales_volatility]
    function test_volatility(settings = Dict())
        success = true
        for f in vol_functions
            k = Symbol(f)
            args = reduce(union, Base.kwarg_decl.(methods(f)))
            releventsettings = intersect(args, keys(settings))
            sets =
                (length(releventsettings) > 0) ? subset_dict(settings, releventsettings) :
                Dict()
            estimate = f(ts1, assets; sets...)
            estimate = (Symbol(f) == :two_scales_volatility) ? estimate[1] : estimate
            estimate_c = estimate_volatility(ts1, assets, k; sets...)
            cor_err = calculate_mean_abs_distance(estimate, estimate_c)
            t1 = all((cor_err .< 10 * eps()) .| isnan.(cor_err))
            if (t1 == false)
                println("In convenience function volatility test there is a error for ", k)
            end
            success = success & (t1)
        end
        return success
    end
    @test test_volatility()
    settings1 = Dict(
        [:rough_guess_number_of_intervals, :num_grids] .=>     [7, 2 * default_num_grids(ts1)],
    )
    @test test_volatility(settings1)



    # Testing Regularisation
    covar = two_scales_covariance(ts1; regularisation = missing)

    reg_functions = [
        identity_regularisation,
        eigenvalue_clean,
        nearest_correlation_matrix,
        nearest_psd_matrix,
    ]
    function test_regularisation(covar, settings = Dict())
        success = true
        for f in reg_functions
            k = Symbol(f)
            args = reduce(union, Base.kwarg_decl.(methods(f)))
            releventsettings = intersect(args, keys(settings))
            sets =
                (length(releventsettings) > 0) ? subset_dict(settings, releventsettings) :
                Dict()
            estimate = f(covar, ts1; sets...)
            estimate_c = regularise(covar, ts1, k; sets...)
            cor_err = calculate_mean_abs_distance(estimate, estimate_c).Correlation_error
            t1 = (cor_err .< 10 * eps()) | isnan(cor_err)
            if (t1 == false)
                println(
                    "In convenience function covariance test there is a covariance error for ",
                    k,
                )
            end
            vol_err = calculate_mean_abs_distance(estimate, estimate_c).Volatility_error
            t2 = (vol_err .< 10 * eps()) | isnan(vol_err)
            if (t2 == false)
                println(
                    "In convenience function covariance test there is a volatility error for ",
                    k,
                )
            end
            success = success & (t1 & t2)
        end
        return success
    end
    @test test_regularisation(covar)
    settings1 = Dict{Symbol,Any}(
        [:apply_to_covariance, :identity_weight, :doDykstra] .=> Any[false, 0.5, false],
    )
    @test test_regularisation(covar, settings1)
end
