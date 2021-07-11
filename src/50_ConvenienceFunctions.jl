"""
estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :TwoScales;
                             num_grids::Real = duration(ts)/10, return_calc::Function = simple_differencing,
                             time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                             use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)

This is a convenience wrapper for the two volatility estimation techniques included in this package.
The method can be :Simple or :TwoScales in which case the simple or two scales volatilty methods will be called.
"""
function estimate_volatility(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts), method::Symbol = :TwoScales;
                             num_grids::Real = duration(ts)/10, return_calc::Function = simple_differencing,
                             time_grid::Union{Missing,Dict} = missing , fixed_spacing::Union{Missing,Dict,<:Real} = missing,
                             use_all_obs::Bool = false, rough_guess_number_of_intervals::Integer = 5)
    if method == :Simple
        return simple_covariance(ts, assets; return_calc = return_calc,
                                   time_grid = time_grid, fixed_spacing = fixed_spacing,
                                   use_all_obs = use_all_obs, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :TwoScales
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
                             regularisation::Union{Missing,Function} = eigenvalue_clean, only_regulise_if_not_PSD::Bool = false,
                             return_calc::Function = simple_differencing, time_grid::Union{Missing,Vector} = missing,
                             fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5, # General Inputs
                             kernel::HFC_Kernel{<:Real} = parzen, H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6, m::Integer = 2, # BNHLS parameters
                             numJ::Integer = 100, num_blocks::Integer = 10, block_width::Real = (maximum(ts.df[:,ts.time]) - minimum(ts.df[:,ts.time])) / num_blocks, microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2], # Spectral Covariance parameters
                             theta::Real = 0.15, g::NamedTuple = g, # Preaveraging
                             equalweight::Bool = false) # Two Scales parameters
    if method == :Simple
        return simple_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                   return_calc = return_calc, time_grid = time_grid,
                                   fixed_spacing = fixed_spacing, refresh_times = refresh_times, rough_guess_number_of_intervals = rough_guess_number_of_intervals)
    elseif method == :BNHLS
        return bnhls_covariance(ts, assets; regularisation = regularisation,
                                  only_regulise_if_not_PSD = only_regulise_if_not_PSD, kernel = kernel, H = H,
                                  m = m, return_calc = return_calc)
    elseif method == :Spectral
        return spectral_covariance(ts, assets; regularisation = regularisation,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, numJ = numJ, num_blocks = num_blocks, block_width = block_width,
                                     microstructure_noise_var = microstructure_noise_var, return_calc = return_calc)
    elseif method == :Preaveraging
        return preaveraged_covariance(ts, assets; regularisation = regularisation,
                                     only_regulise_if_not_PSD = only_regulise_if_not_PSD, theta = theta, g = g, return_calc = return_calc)
    elseif method == :TwoScales
        return two_scales_covariance(ts, assets; regularisation = regularisation, only_regulise_if_not_PSD = only_regulise_if_not_PSD,
                                    equalweight = equalweight, num_grids = num_grids, return_calc = return_calc)
    else
        error("The covariance method chosen must be :Simple, :BNHLS, :Spectral, Preaveraging or :TwoScales")
    end
end

function regularise()

end
