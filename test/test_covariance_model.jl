using Test
using HighFrequencyCovariance
using LinearAlgebra, Random, Dates, DataFrames
function make_covariance_model(whichone)
    if whichone == :simple
        brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.00;
                                          0.0 1.00 0.5 0.25;
                                          0.0 0.00 1.0 0.25;
                                          0.0 0.00 0.0 1.00])
        assets = [:BARC, :HSBC, :VODL, :RYAL]
        twister = MersenneTwister(1)
        time_period_per_unit = Dates.Day(1)
        vols = [0.02,0.03,0.04,0.05]
        cm = CovarianceMatrix(brownian_corr_matrix, vols, assets, time_period_per_unit)
        drifts = [0.0,0.0,0.1,0.1]
        means = [0.05,0.01,0.01,-1.02]
        return CovarianceModel(cm, means, drifts)
    elseif whichone == :zeromean
        brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.00;
                                          0.0 1.00 0.5 0.25;
                                          0.0 0.00 1.0 0.25;
                                          0.0 0.00 0.0 1.00])
        assets = [:BARC, :HSBC, :VODL, :RYAL]
        twister = MersenneTwister(1)
        time_period_per_unit = Dates.Day(1)
        vols = [0.02,0.03,0.04,0.05]
        cm = CovarianceMatrix(brownian_corr_matrix, vols, assets, time_period_per_unit)
        drifts = [0.0,0.0,0.0,0.0]
        means = [0.0,0.0,0.0,0.0]
        return CovarianceModel(cm, means, drifts)
    elseif whichone == :boring
        brownian_corr_matrix = Hermitian([1.0 0.0 0.0 0.0;
                                          0.0 1.0 0.0 0.0;
                                          0.0 0.0 1.0 0.0;
                                          0.0 0.0 0.0 1.0])
        assets = [:VODL, :BARC, :HSBC, :RYAL]
        twister = MersenneTwister(1)
        time_period_per_unit = Dates.Day(1)
        vols = [0.02,0.02,0.02,0.02]
        cm = CovarianceMatrix(brownian_corr_matrix, vols, assets, time_period_per_unit)
        drifts = [0.0,0.0,0.0,0.0]
        means = [0.0,0.0,0.0,0.0]
        return CovarianceModel(cm, means, drifts)
    else
        error("Do not understand this input!")
    end
    error("Unreachable code.")
end
function is_close(x,y)
    return abs(x - y) < 0.00000000001
end

@testset "test basics on covariance model" begin
    cm = make_covariance_model(:simple)
    cm_boring = make_covariance_model(:boring)
    # Testing showing.
    show(cm)
    show(cm, 3, 2)
    # Testing is valid stuff
    @test is_psd_matrix(cm)
    @test valid_correlation_matrix(cm)
    # Test relabelling
    cm_relab = relabel(cm, Dict([:BARC] .=> [:Barclays]))
    @test sum([isnan(x) for x in calculate_mean_abs_distance(cm, cm_relab; return_nans_if_symbols_dont_match = true)]) == 4
    # Testing basic accessor functions
    @test is_close(get_mean(cm, :BARC), 0.05)
    @test is_close(get_mean(cm, :VODL), 0.01)
    @test is_close(get_drift(cm, :VODL, Dates.Day(0)), 0.0)
    @test is_close(get_drift(cm, :VODL, Dates.Day(1)), 0.1)
    @test is_close(get_drift(cm, :VODL, Dates.Day(4)), 0.4)
    @test is_close(get_volatility(cm, :VODL, Dates.Day(0)), 0.0)
    @test is_close(get_volatility(cm, :VODL, Dates.Day(1)), 0.04)
    @test is_close(get_volatility(cm, :VODL, Dates.Day(4)), 0.08)
    @test is_close(get_correlation(cm, :VODL, :BARC), 0.5)
    # Testing Serialisation
    cm_as_df = DataFrames.DataFrame(cm)
    cm_as_df_as_cm = CovarianceModel(cm_as_df)
    cm_as_df2 = DataFrames.DataFrame(cm, Dict(["col1", "col2"] .=> ['A', 'B']))
    cm_as_df_as_cm2 = CovarianceModel(cm_as_df2)
    # Testing distance metrics
    dist = calculate_mean_abs_distance(cm, cm_as_df_as_cm)
    @test all([is_close(dist[x], 0.0) for x in  keys(dist)])
    dist = calculate_mean_abs_distance(cm, cm_as_df_as_cm2)
    @test all([is_close(dist[x], 0.0) for x in  keys(dist)])
    dist2 = calculate_mean_abs_distance(cm, cm_boring)
    @test all([!is_close(dist2[x], 0.0) for x in  keys(dist)])
    @test is_close(calculate_mean_abs_distance_covar(cm, cm_as_df_as_cm), 0.0)
    @test !is_close(calculate_mean_abs_distance_covar(cm, cm_boring), 0.0)
    # Testing making a make_nan_covariance_model
    make_nan_covariance_model([:BARC, :VODL], Dates.Day(2))
    # Testing getting covariance and mean
    covv, meann = covariance_and_mean(cm)
    @test sum([abs(x) > 0.001 for x in meann]) == 4
    covv, meann = covariance_and_mean(cm_boring)
    @test sum([abs(x) > 0.001 for x in meann]) == 0
end


function each_dist(a,b, target)
    dist = calculate_mean_abs_distance(a, b)
    return Bool[is_close(dist[x], target) for x in  keys(dist)]
end


# Testing combinations
@testset "test combinations of covariance model" begin
    cm = make_covariance_model(:simple)
    cm_boring = make_covariance_model(:boring)
    combo = combine_covariance_models([cm_boring, cm], [0.2, 0.8])
    @test !any(each_dist(cm, combo, 0.0))
    @test !any(each_dist(cm_boring, combo, 0.0))
    combo = combine_covariance_models([cm, cm], [0.2, 0.8])
    @test is_close(calculate_mean_abs_distance_covar(cm, combo), 0.0)
    @test !is_close(calculate_mean_abs_distance_covar(cm_boring, combo), 0.0)
    @test all(each_dist(cm, combo, 0.0))
    @test !any(each_dist(cm_boring, combo, 0.0))
    combo = combine_covariance_models([cm, cm], [0.2, 0.8])
    @test is_close(calculate_mean_abs_distance_covar(cm, combo), 0.0)
    @test !is_close(calculate_mean_abs_distance_covar(cm_boring, combo), 0.0)
end

@testset "test conditioning of multivariate Gaussian" begin
    cm = make_covariance_model(:zeromean)
    cond = get_conditional_distribution(cm, [:VODL, :HSBC, :RYAL], Float64[0.004, 0.002, 0.001], Dates.Day(1))
    @test get_mean(cond, :BARC) > 0.001 
    
    cm = make_covariance_model(:boring)
    cond = get_conditional_distribution(cm, [:VODL, :HSBC, :RYAL], Float64[0.004, 0.002, 0.001], Dates.Day(1))
    @test is_close(get_mean(cond, :BARC), 0.0)    
end

@testset "test conditioning of multivariate Gaussian 2" begin
    cm = make_covariance_model(:zeromean)
    draws = DataFrame(get_draws(cm, 100000))
    draws[!,:BARC_cond_exp] = repeat([NaN], nrow(draws))
    draws[!,:BARC_cond_vol] = repeat([NaN], nrow(draws))
    for i in 1:nrow(draws)
        cond = get_conditional_distribution(cm, [:VODL, :HSBC, :RYAL], Vector(draws[i,  [:VODL, :HSBC, :RYAL]]), cm.cm.time_period_per_unit)
        draws[i, :BARC_cond_exp] = get_drift(cond, :BARC, cond.cm.time_period_per_unit) + get_mean(cond, :BARC)
        draws[i, :BARC_cond_vol] = get_volatility(cond, :BARC, cond.cm.time_period_per_unit)
    end
    draws[!, :zz] = (draws[:,:BARC] .- draws[:,:BARC_cond_exp]) ./ draws[:, :BARC_cond_vol]
    @test abs(mean(draws[:,:zz])) < 0.01
    @test abs(std(draws[:,:zz]) - 1.0) < 0.01


    cm = make_covariance_model(:simple)
    draws = DataFrame(get_draws(cm, 100000))
    draws[!,:BARC_cond_exp] = repeat([NaN], nrow(draws))
    draws[!,:BARC_cond_vol] = repeat([NaN], nrow(draws))
    for i in 1:nrow(draws)
        cond = get_conditional_distribution(cm, [:VODL, :HSBC, :RYAL], Vector(draws[i,  [:VODL, :HSBC, :RYAL]]), cm.cm.time_period_per_unit)
        draws[i, :BARC_cond_exp] = get_drift(cond, :BARC, cond.cm.time_period_per_unit) + get_mean(cond, :BARC)
        draws[i, :BARC_cond_vol] = get_volatility(cond, :BARC, cond.cm.time_period_per_unit)
    end
    draws[!, :zz] = (draws[:,:BARC] .- draws[:,:BARC_cond_exp]) ./ draws[:, :BARC_cond_vol]
    @test abs(mean(draws[:,:zz])) < 0.01
    @test abs(std(draws[:,:zz]) - 1.0) < 0.01

    cm = make_covariance_model(:boring)
    draws = DataFrame(get_draws(cm, 100000))
    draws[!,:BARC_cond_exp] = repeat([NaN], nrow(draws))
    draws[!,:BARC_cond_vol] = repeat([NaN], nrow(draws))
    for i in 1:nrow(draws)
        cond = get_conditional_distribution(cm, [:VODL, :HSBC, :RYAL], Vector(draws[i,  [:VODL, :HSBC, :RYAL]]), cm.cm.time_period_per_unit)
        draws[i, :BARC_cond_exp] = get_drift(cond, :BARC, cond.cm.time_period_per_unit) + get_mean(cond, :BARC)
        draws[i, :BARC_cond_vol] = get_volatility(cond, :BARC, cond.cm.time_period_per_unit)
    end
    draws[!, :zz] = (draws[:,:BARC] .- draws[:,:BARC_cond_exp]) ./ draws[:, :BARC_cond_vol]
    @test abs(mean(draws[:,:zz])) < 0.01
    @test abs(std(draws[:,:zz]) - 1.0) < 0.01

end
