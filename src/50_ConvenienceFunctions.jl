"""
This is a convenience wrapper for the two volatility estimation techniques included in this package.
The method can be :simple_volatility or :two_scales_volatility in which case the simple or two scales volatilty methods will be called.
"""
function estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :two_scales_volatility;
                             num_grids::Real = default_num_grids(ts),
                             time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                             use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)
    if method == :simple_volatility
        # simple_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
        #                           time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
        #                           use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)
        return simple_volatility(ts, assets;
                                   time_grid = time_grid, fixed_spacing = fixed_spacing,
                                   use_all_obs = use_all_obs, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :two_scales_volatility
        # two_scales_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts);
        #                num_grids::Real = default_num_grids(ts))
        return two_scales_volatility(ts, assets; num_grids = num_grids)
    else
        error("The volatilty method chosen must be either :simple_volatility or :two_scales_volatility")
    end
end




"""
This is a convenience wrapper for the five covariance estimation techniques included in this package.
The method can be :simple_covariance, :bnhls_covariance, :spectral_covariance, :preaveraged_covariance or :two_scales_covariance.
"""
function estimate_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :preaveraged_covariance;
                             regularisation::Union{Missing,Symbol} = :default, regularisation_params::Dict = Dict(),
                             only_regulise_if_not_PSD::Bool = false,
                             time_grid::Union{Missing,Vector} = missing,
                             fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5, # General Inputs
                             kernel::HFC_Kernel{<:Real} = parzen, H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6, m::Integer = 2, # BNHLS parameters
                             numJ::Integer = 100, num_blocks::Integer = 10, block_width::Real = (maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])) / num_blocks, microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2], # Spectral Covariance parameters
                             theta::Real = 0.15, g::NamedTuple = g, # Preaveraging
                             equalweight::Bool = false, num_grids::Real = default_num_grids(ts)) # Two Scales parameters
    if regularisation == :default
        regularisation = (method == :two_scales_covariance) ? :correlation_default : :covariance_default
    end

    if method == :simple_covariance
        # simple_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Symbol} = :CovarianceDefault, regularisation_params::Dict = Dict(),
        #                           only_regulise_if_not_PSD::Bool = false, time_grid::Union{Missing,Vector} = missing,
        #                           fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5)
        return simple_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                   time_grid = time_grid,
                                   fixed_spacing = fixed_spacing, refresh_times = refresh_times, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :bnhls_covariance
        # bnhls_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Symbol} = :CovarianceDefault, regularisation_params::Dict = Dict(),
        #                          only_regulise_if_not_PSD::Bool = false, kernel::HFC_Kernel{<:Real} = parzen, H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6,
        #                          m::Integer = 2)
        return bnhls_covariance(ts, assets; regularisation = regularisation,
                                  only_regulise_if_not_PSD = only_regulise_if_not_PSD, kernel = kernel, H = H,
                                  m = m)
    elseif method == :spectral_covariance
        # spectral_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Symbol} = :CovarianceDefault, regularisation_params::Dict = Dict(),
        #                             only_regulise_if_not_PSD::Bool = false, numJ::Integer = 100, num_blocks::Integer = 10, block_width::Real = (maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])) / num_blocks,
        #                             microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2])
        return spectral_covariance(ts, assets; regularisation = regularisation,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, numJ = numJ, num_blocks = num_blocks, block_width = block_width,
                                     microstructure_noise_var = microstructure_noise_var)
    elseif method == :preaveraged_covariance
        # preaveraged_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Symbol} = :CovarianceDefault, regularisation_params::Dict = Dict(),
        #                             only_regulise_if_not_PSD::Bool = false, theta::Real = 0.15, g::NamedTuple = g)
        return preaveraged_covariance(ts, assets; regularisation = regularisation,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, theta = theta, g = g)
    elseif method == :two_scales_covariance
        # two_scales_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Symbol} = :CorrelationDefault, regularisation_params::Dict = Dict(),
        #                             only_regulise_if_not_PSD::Bool = false, equalweight::Bool = false, num_grids::Real = default_num_grids(ts))
        return two_scales_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                    equalweight = equalweight, num_grids = num_grids)
    else
        error("The covariance method chosen must be :simple_covariance, :bnhls_covariance, :spectral_covariance, :preaveraged_covariance or :two_scales_covariance")
    end
end

# The Hermitian version
"""
This is a convenience wrapper for the regularisation techniques.
    The methods can be:
    :identity_regularisation, :eigenvalue_clean, :nearest_correlation_matrix or :nearest_psd_matrix. You can also choose :covariance_default (which is :nearest_psd_matrix) or  :correlation_default (which is :nearest_correlation_matrix).
"""
function regularise(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector, method::Symbol = :correlation_default;
                    identity_weight::Union{Missing,<:Real} = missing, spacing::Union{Missing,<:Real} = missing,
                    weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
                    doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
    if method == :covariance_default
        method = :nearest_psd_matrix
    elseif method == :correlation_default
        method = :nearest_correlation_matrix
    end

    if method == :identity_regularisation
        # identity_regularisation(mat::Hermitian, ts::SortedDataFrame,  mat_labels::Vector; identity_weight::Union{Missing,<:Real} = missing, spacing::Union{Missing,<:Real} = missing)
        return identity_regularisation(mat, ts,  mat_labels; identity_weight = identity_weight, spacing = spacing)
    elseif method == :eigenvalue_clean
        # eigenvalue_clean(mat::Hermitian, ts::SortedDataFrame, mat_labels = missing)
        return eigenvalue_clean(mat, ts, mat_labels)
    elseif method == :nearest_correlation_matrix
        # nearest_correlation_matrix(mat::Hermitian, mat_labels::Vector = missing; weighting_matrix = Diagonal(eltype(mat).(I(size(mat)[1]))),
        #                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
        return nearest_correlation_matrix(mat, mat_labels; weighting_matrix = weighting_matrix, doDykstra = doDykstra,
                                          stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    elseif method == :nearest_psd_matrix
        # nearest_psd_matrix(mat::Hermitian)
        return nearest_psd_matrix(mat)
    else
        error("The covariance method chosen must be :identity_regularisation, :eigenvalue_clean, :nearest_correlation_matrix or :nearest_psd_matrix. You can also choose :covariance_default (which is :nearest_psd_matrix) or  :correlation_default (which is :nearest_correlation_matrix).")
    end
end


# The CovarianceMatrix version
"""
This is a convenience wrapper for the regularisation techniques.
    The methods can be:
    :identity_regularisation, :eigenvalue_clean, :nearest_correlation_matrix or :nearest_psd_matrix. You can also choose :covariance_default (which is :nearest_psd_matrix) or  :correlation_default (which is :nearest_correlation_matrix).
"""
function regularise(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame, method::Symbol = :nearest_correlation_matrix;
                    identity_weight::Union{Missing,<:Real} = missing, spacing::Union{Missing,<:Real} = missing,
                    apply_to_covariance::Bool = true,
                    weighting_matrix = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
                    doDykstra = true, stop_at_first_correlation_matrix = true, max_iterates = 1000)
    if method == :covariance_default
        method = :nearest_psd_matrix
    elseif method == :correlation_default
        method = :nearest_correlation_matrix
    end

    if method == :identity_regularisation
        # identity_regularisation(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; identity_weight::Union{Missing,<:Real} = missing,
        #                         spacing::Union{Missing,<:Real} = missing, apply_to_covariance::Bool = true)
        return identity_regularisation(covariance_matrix, ts; identity_weight = identity_weight, spacing = spacing, apply_to_covariance = apply_to_covariance)
    elseif method == :eigenvalue_clean
        # eigenvalue_clean(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; apply_to_covariance::Bool = true)
        return eigenvalue_clean(covariance_matrix, ts; apply_to_covariance = apply_to_covariance)
    elseif method == :nearest_correlation_matrix
        # nearest_correlation_matrix(covariance_matrix::CovarianceMatrix, ts::SortedDataFrame; weighting_matrix::Union{Diagonal,Hermitian} = Diagonal(eltype(covariance_matrix.correlation).(I(size(covariance_matrix.correlation)[1]))),
        #                             doDykstra::Bool = true, stop_at_first_correlation_matrix::Bool = true, max_iterates::Integer = 1000)
        return nearest_correlation_matrix(covariance_matrix, ts; weighting_matrix = weighting_matrix, doDykstra = doDykstra,
                                          stop_at_first_correlation_matrix = stop_at_first_correlation_matrix, max_iterates = max_iterates)
    elseif method == :nearest_psd_matrix
        # nearest_psd_matrix(covariance_matrix::CovarianceMatrix; apply_to_covariance::Bool = true)
        return nearest_psd_matrix(covariance_matrix; apply_to_covariance = apply_to_covariance)
    else
        error("The covariance method chosen must be :identity_regularisation, :eigenvalue_clean, :nearest_correlation_matrix or :nearest_psd_matrix. You can also choose :covariance_default (which is :nearest_psd_matrix) or  :correlation_default (which is :nearest_correlation_matrix).")
    end
end