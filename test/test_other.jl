using Revise
using UnivariateFunctions
using StochasticIntegrals
using DataFrames
using Dates
using LinearAlgebra
using Distributions: Exponential, MersenneTwister, quantile
using DataStructures: OrderedDict
using Statistics: std, var, mean, cov
using HighFrequencyCovariance
using Sobol
using Random
const tol = 10*eps()

brownian_corr_matrix = Hermitian([1.0 0.75 0.5 0.0;
                                  0.0 1.0 0.5 0.25;
                                  0.0 0.0 1.0 0.25;
                                  0.0 0.0 0.0 1.0])
brownian_ids = [:BARC, :HSBC, :VODL, :RYAL]

BARC  = ItoIntegral(:BARC, PE_Function(0.02))
HSBC  = ItoIntegral(:HSBC, PE_Function(0.03))
VODL  = ItoIntegral(:VODL, PE_Function(0.04))
RYAL  = ItoIntegral(:RYAL, PE_Function(0.05))
ito_integrals = Dict([:BARC, :HSBC, :VODL, :RYAL] .=> [BARC, HSBC, VODL, RYAL])
ito_set_ = ItoSet(brownian_corr_matrix, brownian_ids, ito_integrals)

covar = ForwardCovariance(ito_set_, 0.0, 1.0)
stock_processes = Dict([:BARC, :HSBC, :VODL, :RYAL] .=>
                           [ItoProcess(0.0, 180.0, PE_Function(0.00), ito_integrals[:BARC]),
                           ItoProcess(0.0, 360.0, PE_Function(0.00), ito_integrals[:HSBC]),
                           ItoProcess(0.0, 720.0, PE_Function(0.00), ito_integrals[:VODL]),
                           ItoProcess(0.0, 500.0, PE_Function(0.0), ito_integrals[:RYAL])])

# The asyncronous case
update_rates = OrderedDict([:BARC, :HSBC, :VODL, :RYAL] .=> [Exponential(2.0), Exponential(3.0), Exponential(15.0), Exponential(2.5)])
ts = make_ito_process_non_syncronous_time_series(stock_processes, covar, update_rates, 10000; timing_twister = MersenneTwister(2), ito_twister = MersenneTwister(5))
ts = SortedDataFrame(ts)

naive_covar = simple_covariance(ts, brownian_ids)
realised_covar = bnhls_covariance(ts, brownian_ids)
preav_covar = preaveraged_covariance(ts, brownian_ids)








# Testing the next tick function
\zauld return something
ismissing(next_tick(ts, ts.df[nrow(ts.df)-2,ts.time]) )  # Just before last tick
ismissing(next_tick(ts, ts.df[nrow(ts.df),ts.time] + 1.0)) # After last tick

refresh_frequency = secs_between_refreshes(ts)

C = combine_covariance_matrices(A, B, 0.4)



obs = nrow(dd)
S = eigenvalue_clean(mat; obs = obs)
# In: ts, blocks
refresh_frequency = secs_between_refreshes(ts)
at_times = get_all_refresh_times(ts)
dd_compiled = latest_value(ts, at_times)
dd = get_returns(dd_compiled, 1.0; returns = :simple)

identity_regularisation(dd, mat, mat_labels)

R = Float64


A = mat[1:3,1:3]
A_labels = mat_labels[1:3]
B =  2 .* mat[3:4,3:4]
B_labels = mat_labels[3:4]
A_mixture = 0.8
combined_matrix, combined_labels = combine_labelled_Hermitian_matrices(A, A_labels, B, B_labels, A_mixture)
ismissing(combined_matrix[1,4])
combined_matrix[1,1] == A[1,1]
combined_matrix[4,4] == B[2,2]
abs(combined_matrix[3,3] - (A[3,3] * (0.8) + 0.2 * B[1,1])) < 1e-12

blocking_dd = DataFrame(sets = [Set([:BARC, :HSBC, :VODL, :RYAL]), Set([:BARC, :RYAL])], functions = [:A, :A], options = [:A, :A])
functions = Dict{Symbol,Function}(:A => syncronised_naive_eigen)
options = Dict{Symbol,Dict}(:A => Dict(:returns_interval => 1.0))

regularisation_options = Dict(:obs => 1000)
bwe, labs   = blockwise_estimation(ts, blocking_dd, :eigen; functions = functions, options = options, regularisation_options = regularisation_options)
bwe1, labs1 = blockwise_estimation(ts, DataFrame(blocking_dd[1,:]), :eigen; functions = functions, options = options, regularisation_options = regularisation_options)

returns = Matrix(dd[:,labs])

functions = Dict{Symbol,Function}(:A => syncronised_kernel_eigen)
bwe, labs   = blockwise_estimation(ts, blocking_dd, :eigen; functions = functions, options = options, regularisation_options = regularisation_options)
bwe1, labs1 = blockwise_estimation(ts, DataFrame(blocking_dd[1,:]), :eigen; functions = functions, options = options, regularisation_options = regularisation_options)

obs_multiple_for_new_block = 0.2

bwe, labs = blockwise_estimation(ts, obs_multiple_for_new_block, syncronised_naive_eigen, :eigen)
