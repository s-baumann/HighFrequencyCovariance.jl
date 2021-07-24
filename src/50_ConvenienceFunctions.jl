"""
estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :TwoScales;
                             num_grids::Real = duration(ts)/10, return_calc::Function = simple_differencing,
                             time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                             use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)

This is a convenience wrapper for the two volatility estimation techniques included in this package.
The method can be :Simple or :TwoScales in which case the simple or two scales volatilty methods will be called.
"""
function estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :TwoScales;
                             num_grids::Real = default_num_grids(ts), return_calc::Function = simple_differencing,
                             time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                             use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)
    if method == :Simple
        # simple_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); return_calc::Function = simple_differencing,
        #                           time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
        #                           use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)
        return simple_volatility(ts, assets; return_calc = return_calc,
                                   time_grid = time_grid, fixed_spacing = fixed_spacing,
                                   use_all_obs = use_all_obs, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :TwoScales
        # two_scales_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
        #                num_grids::Real = default_num_grids(ts), return_calc::Function = simple_differencing)
        return two_scales_volatility(ts, assets; num_grids = num_grids, return_calc = return_calc)
    else
        error("The volatilty method chosen must be either :Simple or :TwoScales")
    end
end




"""
estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :TwoScales;
                             num_grids::Real = duration(ts)/10, return_calc::Function = simple_differencing,
                             time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                             use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)

This is a convenience wrapper for the two volatility estimation techniques included in this package.
The method can be :Simple or :TwoScales in which case the simple or two scales volatilty methods will be called.
"""
function estimate_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :Preaveraging;
                             regularisation::Union{Missing,Function} = nearest_correlation_matrix, only_regulise_if_not_PSD::Bool = false,
                             return_calc::Function = simple_differencing, time_grid::Union{Missing,Vector} = missing,
                             fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5, # General Inputs
                             kernel::HFC_Kernel{<:Real} = parzen, H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6, m::Integer = 2, # BNHLS parameters
                             numJ::Integer = 100, num_blocks::Integer = 10, block_width::Real = (maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])) / num_blocks, microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2], # Spectral Covariance parameters
                             theta::Real = 0.15, g::NamedTuple = g, # Preaveraging
                             equalweight::Bool = false, num_grids::Real = default_num_grids(ts)) # Two Scales parameters
    if method == :Simple
        # simple_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = nearest_correlation_matrix, only_regulise_if_not_PSD::Bool = false,
        #                           return_calc::Function = simple_differencing, time_grid::Union{Missing,Vector} = missing,
        #                           fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5)
        return simple_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                   return_calc = return_calc, time_grid = time_grid,
                                   fixed_spacing = fixed_spacing, refresh_times = refresh_times, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :BNHLS
        # bnhls_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = nearest_correlation_matrix,
        #                          only_regulise_if_not_PSD::Bool = false, kernel::HFC_Kernel{<:Real} = parzen, H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6,
        #                          m::Integer = 2, return_calc::Function = simple_differencing)
        return bnhls_covariance(ts, assets; regularisation = regularisation,
                                  only_regulise_if_not_PSD = only_regulise_if_not_PSD, kernel = kernel, H = H,
                                  m = m, return_calc = return_calc)
    elseif method == :Spectral
        # spectral_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = nearest_correlation_matrix,
        #                             only_regulise_if_not_PSD::Bool = false, numJ::Integer = 100, num_blocks::Integer = 10, block_width::Real = (maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])) / num_blocks,
        #                             microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2], return_calc::Function = simple_differencing)
        return spectral_covariance(ts, assets; regularisation = regularisation,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, numJ = numJ, num_blocks = num_blocks, block_width = block_width,
                                     microstructure_noise_var = microstructure_noise_var, return_calc = return_calc)
    elseif method == :Preaveraging
        # preaveraged_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = nearest_correlation_matrix,
        #                             only_regulise_if_not_PSD::Bool = false, theta::Real = 0.15, g::NamedTuple = g, return_calc::Function = simple_differencing)
        return preaveraged_covariance(ts, assets; regularisation = regularisation,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, theta = theta, g = g, return_calc = return_calc)
    elseif method == :TwoScales
        # two_scales_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = nearest_correlation_matrix,
        #                             only_regulise_if_not_PSD::Bool = false, equalweight::Bool = false, num_grids::Real = default_num_grids(ts), return_calc::Function = simple_differencing)
        return two_scales_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                    equalweight = equalweight, num_grids = num_grids, return_calc = return_calc)
    else
        error("The covariance method chosen must be :Simple, :BNHLS, :Spectral, :Preaveraging or :TwoScales")
    end
end

# The Hermitian version
function regularise(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector, method::Symbol = :NearestCorrelation;
                    identity_weight::Union{Missing,<:Real} = missing, spacing::Union{Missing,<:Real} = missing,
                    return_calc::Function = simple_differencing, weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
                    doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
    if method == :Identity
        # identity_regularisation(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector; identity_weight::Union{Missing,<:Real} = missing, spacing::Union{Missing,<:Real} = missing, return_calc::Function = simple_differencing)
        return identity_regularisation(mat, ts,  mat_labels; identity_weight = identity_weight, spacing = spacing, return_calc = return_calc)
    elseif method == :EigenClean
        # eigenvalue_clean(mat::Hermitian, ts::SortedDataFrame, mat_labels = missing)
        return eigenvalue_clean(mat, ts, mat_labels)
    elseif method == :NearestCorrelation
        # nearest_correlation_matrix(mat::Hermitian, mat_labels::Vector = missing; weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
        #                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
        return nearest_correlation_matrix(mat, mat_labels; weighting_matrix = weighting_matrix, doDykstra = doDykstra,
                                          stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    elseif method == :NearestPSD
        # nearest_psd_matrix(mat::Hermitian)
        return nearest_psd_matrix(mat)
    else
        error("The covariance method chosen must be :Identity, :EigenClean, :NearestCorrelation or :NearestPSD")
    end
end


# The CovarianceMatrix version
function regularise(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame, method::Symbol = :NearestCorrelation;
                    identity_weight::Union{Missing,<:Real} = missing, spacing::Union{Missing,<:Real} = missing,
                    return_calc::Function = simple_differencing, apply_to_covariance::Bool = true,
                    weighting_matrix = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                    doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
    if method == :Identity
        # identity_regularisation(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; identity_weight::Union{Missing,<:Real} = missing,
        #                         spacing::Union{Missing,<:Real} = missing, return_calc::Function = simple_differencing, apply_to_covariance::Bool = true)
        return identity_regularisation(covariance_matrix, ts; identity_weight = identity_weight, spacing = spacing, return_calc = return_calc, apply_to_covariance = apply_to_covariance)
    elseif method == :EigenClean
        # eigenvalue_clean(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; apply_to_covariance::Bool = true)
        return eigenvalue_clean(covariance_matrix, ts; apply_to_covariance = apply_to_covariance)
    elseif method == :NearestCorrelation
        # nearest_correlation_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
        #                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
        return nearest_correlation_matrix(covariance_matrix, ts; weighting_matrix = weighting_matrix, doDykstra = doDykstra,
                                          stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    elseif method == :NearestPSD
        # nearest_psd_matrix(covariance_matrix::CovarianceMatrix; apply_to_covariance::Bool = true)
        return nearest_psd_matrix(covariance_matrix; apply_to_covariance = apply_to_covariance)
    else
        error("The covariance method chosen must be :Identity, :EigenClean, :NearestCorrelation or :NearestPSD")
    end
end
