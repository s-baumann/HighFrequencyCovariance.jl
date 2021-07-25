
"""
Estimation of the covariance matrix in the standard simple way given returns.
https://en.wikipedia.org/wiki/Sample_mean_and_covariance
"""
function simple_covariance_given_returns(returns::Array{R,2}) where R<:Real
    N = size(returns)[2]
    mat = zeros(N,N)
    for i in 1:N
        for j in i:N
            if i == j
                mat[i,i] = var(returns[:,i])
            else
                mat[i,j] = cov(returns[:,i], returns[:,j])
            end
        end
    end
    return Hermitian(mat)
end

"""
Estimation of the covariance matrix in the standard simple way given a time grid.
https://en.wikipedia.org/wiki/Sample_mean_and_covariance
"""
function simple_covariance_given_time_grid(ts::SortedDataFrame, assets::Vector{Symbol}, time_grid::Vector; regularisation::Symbol = :CovarianceDefault,
                                           regularisation_params::Dict = Dict(), only_regulise_if_not_PSD::Bool = false)
    dd_compiled = latest_value(ts, time_grid; assets = assets)
    dd = get_returns(dd_compiled; rescale_for_duration = false)

    if nrow(dd) < 1 return make_nan_covariance_matrix(assets) end

    returns = Matrix(dd[:, assets])
    covariance = simple_covariance_given_returns(returns)

    # Regularisation
    dont_regulise = ismissing(regularisation) || (only_regulise_if_not_PSD && is_psd_matrix(covariance))
    covariance = dont_regulise ? covariance : regularise(covariance, ts, assets, regularisation; regularisation_params... )

    # Packing into a CovarianceMatrix and returning.
    cor, vols = cov2cor_and_vol(covariance, duration(ts))
    return CovarianceMatrix(cor, vols, assets)
end

"""
Estimation of the covariance matrix in the standard simple way.
https://en.wikipedia.org/wiki/Sample_mean_and_covariance
"""
function simple_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Symbol} = :CovarianceDefault, regularisation_params::Dict = Dict(),
                           only_regulise_if_not_PSD::Bool = false, time_grid::Union{Missing,Vector} = missing,
                           fixed_spacing::Union{Missing,<:Real} = missing, refresh_times::Bool = false, rough_guess_number_of_intervals::Integer = 5)
   if ismissing(time_grid)
       time_grid = Vector{eltype(ts.df[:,ts.time])}()
       if refresh_times
           time_grid = get_all_refresh_times(ts, assets)
       elseif !ismissing(fixed_spacing)
            time_grid = collect(minimum(ts.df[:,ts.time]):fixed_spacing:maximum(ts.df[:,ts.time]))
       else
           n_grid = default_spacing(ts; rough_guess_number_of_intervals = rough_guess_number_of_intervals)
           vals = collect(values(n_grid))
           spacing = mean(vals[(isnan.(vals) .== false) .& (isinf.(vals) .== false)])
           spacing = isnan(spacing) ? duration(ts)/20 : spacing
           time_grid = collect(minimum(ts.df[:,ts.time]):spacing:maximum(ts.df[:,ts.time]))
       end
   end
   return simple_covariance_given_time_grid(ts, assets, time_grid; regularisation = regularisation, regularisation_params = regularisation_params, only_regulise_if_not_PSD = only_regulise_if_not_PSD)
end
