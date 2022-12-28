struct CovarianceModel{R<:Real}
    cm::HighFrequencyCovariance.CovarianceMatrix{R}
    means::Vector{R}
    drifts::Vector{R}
end

function Base.show(cm::CovarianceModel)
    println()
    flat_labels = reshape(cm.cm.labels, (1, length(cm.cm.labels)))
    println("Means ")
    Base.print_matrix(stdout, vcat(flat_labels, cm.means'))
    println()
    println("Drifts per time interval of ", cm.cm.time_period_per_unit)
    Base.print_matrix(stdout, vcat(flat_labels, cm.drifts'))
    show(cm.cm)
end
function Base.show(
    cm::CovarianceMatrix,
    sig_figs_volatility::Integer,
    decimal_places_correlation::Integer,
)
    rounded_vols = round.(cm.cm.volatility, sigdigits = sig_figs_volatility)
    rounded_correl =
        Hermitian(round.(cm.cm.correlation, digits = decimal_places_correlation))
    cm_rounded = CovarianceMatrix(rounded_correl, rounded_vols, cm.cm.labels, cm.cm.time_period_per_unit)
    rounded_drifts = round.(cm.drift, sigdigits = sig_figs_volatility)
    rounded_means = round.(cm.mean, sigdigits = sig_figs_volatility)
    show(CovarianceModel(cm_rounded, rounded_means, rounded_drifts))
end

"""
convert_drift(
    drift::Union{Vector{<:Real},Real},
    drift_period::Dates.Period,
    new_drift_period::Dates.Period,
    )

    convert_drift(drift::Missing, drift_period::Dates.Period, new_drift_period::Dates.Period)

### Inputs
* `drift` - The drift you want to convert.
* `drift_period` - The period of the drift you have.
* `new_drift_period` - The period that you want to convert the drift to.
"""
function convert_drift(
    drift::Union{Vector{<:Real},Real},
    drift_period::Dates.Period,
    new_drift_period::Dates.Period,
)
    return drift .* time_period_ratio(new_drift_period, drift_period)
end
convert_vol(drift::Missing, drift_period::Dates.Period, new_drift_period::Dates.Period) = missing

"""
    make_nan_covariance_model(
        labels::Vector{Symbol},
        time_period_per_unit::Dates.Period,
    )

This makes an empty `CovarianceModel` struct with all volatilities, correlations, drifts and means being NaNs.
### Inputs
* `labels` - The names of the asset names for this (empty) `CovarianceMatrix`.
* `time_period_per_unit` - The time interval the volatilities will be for.
### Returns
* An (empty) `CovarianceMatrix`
"""
function make_nan_covariance_model(
    labels::Vector{Symbol},
    time_period_per_unit::Dates.Period,
)
    cmat = make_nan_covariance_matrix(
        labels::Vector{Symbol},
        time_period_per_unit::Dates.Period,
    )
    drifts =  repeat([NaN],length(labels))
    means = repeat([NaN],length(labels))
    return CovarianceModel(cmat, means, drifts)
end



"""
    calculate_mean_abs_distance(
        cov1::CovarianceModel,
        cov2::CovarianceModel,
        decimal_places::Integer = 8;
        return_nans_if_symbols_dont_match::Bool = true,
    )

Calculates the mean absolute distance (elementwise in L1 norm) between two `CovarianceModel`s.
Undefined if any labels differ between the two `CovarianceModel`s.
### Inputs
* `cov1` - The first `CovarianceModel`
* `cov2` - The second `CovarianceModel`
* `decimal_places` - How many decimal places to show the result to.
* `return_nans_if_symbols_dont_match` - If the symbols don't match should it be an error. Or if false we only compare common symbols in both `CovarianceMatrix`s
### Returns
* An `Tuple` with the distance for correlations in first entry and distance for volatilities in the second.
"""
function calculate_mean_abs_distance(
    cov1::CovarianceModel,
    cov2::CovarianceModel,
    decimal_places::Integer = 8;
    return_nans_if_symbols_dont_match::Bool = true,
)
    if return_nans_if_symbols_dont_match && (length(symdiff(cov1.cm.labels, cov2.cm.labels)) != 0)
        return (Correlation_error = NaN, Volatility_error = NaN, drift_error = NaN, mean_error= NaN)
    end
    labels = intersect(cov1.cm.labels, cov2.cm.labels)
    N = length(labels)
    cov11 = rearrange(cov1, labels)
    cov22 = rearrange(cov2, labels)
    cor_error = round(
        sum(abs, cov11.cm.correlation .- cov22.cm.correlation) / ((N^2 - N) / 2),
        digits = decimal_places,
    )
    vol_error =
        round(mean(abs, cov11.cm.volatility .- cov22.cm.volatility), digits = decimal_places)
    drift_error =
        round(mean(abs, cov11.drifts .- cov22.drifts), digits = decimal_places)
    mean_error =
        round(mean(abs, cov11.means .- cov22.means), digits = decimal_places)

    return (Correlation_error = cor_error, Volatility_error = vol_error, drift_error = drift_error, mean_error = mean_error)
end

"""
    calculate_mean_abs_distance_covar(
        cov1::CovarianceMatrix,
        cov2::CovarianceMatrix,
        decimal_places::Integer = 8;
        return_nans_if_symbols_dont_match::Bool = true,
    )


Calculates the mean absolute distance (elementwise in L1 norm) between the covariance matrices (over the natural time unit) of two `CovarianceMatrix`s.
Undefined if any labels differ between the two `CovarianceMatrix`s.

Note that this is different from calculate_mean_abs_distance in that this function returns one real for the distance between actual covariance matrices
rather than a tuple showing the distances in terms of correlation and volatility.

### Inputs
* `cov1` - The first `CovarianceMatrix`
* `cov2` - The second `CovarianceMatrix`
* `decimal_places` - How many decimal places to show the result to.
* `return_nans_if_symbols_dont_match` - If the symbols don't match should it be an error. Or if false we only compare common symbols in both `CovarianceMatrix`s
### Returns
* A Real
"""
function calculate_mean_abs_distance_covar(
    cov1::CovarianceModel,
    cov2::CovarianceModel,
    decimal_places::Integer = 8;
    return_nans_if_symbols_dont_match::Bool = true,
)
    return calculate_mean_abs_distance_covar(cov1.cm, cov2.cm, decimal_places; return_nans_if_symbols_dont_match = return_nans_if_symbols_dont_match)
end


"""
    get_correlation(covar::CovarianceModel, asset1::Symbol, asset2::Symbol)

Extract the correlation between two assets stored in a CovarianceModel.
### Inputs
* `covar` - A `CovarianceModel`
* `asset1` - A `Symbol` representing an asset.
* `asset2` - A `Symbol` representing an asset.
### Returns
* A Scalar (the correlation coefficient).
"""
function get_correlation(covar::CovarianceModel, asset1::Symbol, asset2::Symbol)
    index1 = findfirst(asset1 .== covar.cm.labels)
    index2 = findfirst(asset2 .== covar.cm.labels)
    if isnothing(index1) || isnothing(index2)
        return missing
    end
    return covar.cm.correlation[index1, index2]
end

"""
    get_volatility(
        covar::CovarianceModel,
        asset1::Symbol,
        time_period_per_unit::Dates.Period = covar.time_period_per_unit,
    )

Get the volatility for a stock from a `CovarianceModel`.
### Inputs
* `covar` - A `CovarianceModel`
* `asset1` - A `Symbol` representing an asset.
* `time_period_per_unit` - The time interval the volatilities will be for.
### Returns
* A Scalar (the volatility).
"""
function get_volatility(
    covar::CovarianceModel,
    asset1::Symbol,
    time_period_per_unit::Dates.Period = covar.cm.time_period_per_unit,
)
    index1 = findfirst(asset1 .== covar.cm.labels)
    if isnothing(index1)
        return missing
    end
    return convert_vol(
        covar.cm.volatility[index1],
        covar.cm.time_period_per_unit,
        time_period_per_unit,
    )
end

function get_mean(
    covar::CovarianceModel,
    asset1::Symbol
)
    index1 = findfirst(asset1 .== covar.cm.labels)
    if isnothing(index1)
        return missing
    end
    return covar.means[index1]
end

function get_drift(
    covar::CovarianceModel,
    asset1::Symbol,
    time_period_per_unit::Dates.Period = covar.cm.time_period_per_unit,
)
    index1 = findfirst(asset1 .== covar.cm.labels)
    if isnothing(index1)
        return missing
    end
    return convert_drift(covar.drifts[index1], covar.cm.time_period_per_unit, time_period_per_unit)
end

"""
    is_psd_matrix(covar::CovarianceModel)

Test if a matrix is psd (Positive Semi-Definite). This is done by seeing if all eigenvalues are positive.
If a `Hermitian` is input then it will be tested. If a `CovarianceModel` is input then its correlation matrix will be tested.
### Inputs
* `mat` - A `CovarianceModel`
* `min_eigen_threshold` - How big does the smallest eigenvalue have to be.
### Returns
* A `Bool` that is true if mat is psd and false if not.
"""
is_psd_matrix(covar::CovarianceModel) = is_psd_matrix(covar.cm.correlation)

"""
    valid_correlation_matrix(covar::CovarianceModel, min_eigen_threshold::Real = 0.0)
"""
valid_correlation_matrix(covar::CovarianceModel, min_eigen_threshold::Real = 0.0) =
    valid_correlation_matrix(covar.cm.correlation, min_eigen_threshold)

"""
    relabel(covar::CovarianceModel, mapping::Dict{Symbol,Symbol})

This relabels a CovarianceModel struct to give all the assets alternative names.
### Inputs
* `covar` - The `CovarianceModel` object you want to relabel.
* `mapping` - A dict mapping from the names you have to the names you want.
### Returns
* A `CovarianceModel` the same as the one you input but with new labels.
"""
function relabel(covar::CovarianceModel, mapping::Dict{Symbol,Symbol})
    new_cm = relabel(covar.cm, mapping)
    return CovarianceModel(new_cm, covar.drifts, covar.means)
end


"""
    covariance(
        cm::CovarianceMatrix,
        period::Dates.Period = cm.time_period_per_unit,
        assets::Vector{Symbol} = cm.labels,
    )

This makes a `Hermitian` matrix for the covariance matrix over some duration.
### Inputs
* `cm` - A `CovarianceMatrix` struct.
* `period` - A duration for which you want a covariance matrix. This should be in a Dates.Period.
* `assets` - What assets in include in the covariance matrix.
### Returns
* A `Hermitian`. The labelling of assets for each row/column is as per the input `assets` vector.
"""
function covariance_and_mean(
    cm::CovarianceModel,
    period::Dates.Period = cm.time_period_per_unit,
    assets::Vector{Symbol} = cm.labels,
)
    cm2 = rearrange(cm.cm, assets)
    sds = convert_vol(cm2.cm.volatility, cm2.cm.time_period_per_unit, period)
    drifts_to_time = convert_drift(cm2.drifts, cm2.cm.time_period_per_unit, period)
    newcor = cor_to_cov(cm2.cm.correlation, sds)
    new_mean = cm2.means + drifts_to_time
    return newcor, new_mean
end

"""
    combine_covariance_models(
        vect::Vector{CovarianceModel{T}},
        cor_weights::Vector{<:Real} = repeat([1.0], length(vect)),
        vol_weights::Vector{<:Real} = cor_weights,
        time_period_per_unit::Union{Missing,Dates.Period} = vect[1].time_period_per_unit,
    ) where T<:Real

Combines a vector of `CovarianceModel` structs into one `CovarianceModel` struct.
### Inputs
* `vect` - A vector of `CovarianceModel` structs.
* `cor_weights` - A vector for how much to weight the correlations from each covariance matrix (by default they will be equalweighted).
* `vol_weights` - A vector for how much to weight the volatilities from each covariance matrix (by default they will be equalweighted).
* `time_period_per_unit` - What time period should the volatilities be scaled to.
### Returns
* A matrix and a vector of labels for each row/column of the matrix.
"""
function combine_covariance_models(
    vect::Vector{CovarianceModel{T}},
    cor_weights::Vector{<:Real} = repeat([1.0], length(vect)),
    vol_weights::Vector{<:Real} = cor_weights,
    drift_weights::Vector{<:Real} = cor_weights,
    mean_weights::Vector{<:Real} = cor_weights,
    time_period_per_unit::Union{Missing,Dates.Period} = vect[1].cm.time_period_per_unit,
) where T<:Real
    combo = combine_covariance_matrices(
        [x.cm for x in vect],
        cor_weights, vol_weights, time_period_per_unit)
    

    if length(vect) < 1
        error("An empty vector of covariance matrices was input. So not possible to combine.")
    end
    all_labels = combo.labels
    dims = length(all_labels)
    R = promote_type(map(x -> eltype(x.cm.correlation), vect)...)
    new_means = Array{R,1}(undef, dims)
    new_drifts = Array{R,1}(undef, dims)
    for row = 1:dims
        row_label = all_labels[row]
        # Means
        means = map(i -> get_mean(vect[i], row_label), 1:length(vect) )
        valid_entries = setdiff(1:length(means), findall(is_missing_nan_inf.(means)))
        new_means[row] = weighted_mean(means[valid_entries], mean_weights[valid_entries])
        # Drifts
        drifts = map(
            i -> convert_drift(
                get_drift(vect[i], row_label),
                vect[i].cm.time_period_per_unit,
                time_period_per_unit,
            ),
            1:length(vect),
        )
        valid_entries = setdiff(1:length(drifts), findall(is_missing_nan_inf.(drifts)))
        new_drifts[row] = weighted_mean(drifts[valid_entries], drift_weights[valid_entries])
    end
    return CovarianceModel(combo, new_means, new_drifts)
end

"""
    rearrange(
        cm::CovarianceMatrix,
        labels::Vector{Symbol},
        time_period_per_unit::Union{Missing,Dates.Period} = cm.time_period_per_unit,
    )

Rearrange the order of labels in a `CovarianceMatrix`.
### Takes
* `cm` - A `CovarianceMatrix`.
* `labels` - A `Vector` of labels.
* `time_period_per_unit` - The time period you want for the resultant Covariance Matrix
### Returns
* A `CovarianceMatrix`.
"""
function rearrange(
    cm::CovarianceModel,
    labels::Vector{Symbol},
    time_period_per_unit::Union{Missing,Dates.Period} = cm.cm.time_period_per_unit,
)
    if length(setdiff(labels, cm.cm.labels)) > 0
        error("You put in labels that are not in the CovarianceMatrix")
    end
    same_assets = (length(cm.cm.labels) == length(labels)) && (all(cm.cm.labels .== labels))
    same_period = (time_period_per_unit == cm.cm.time_period_per_unit)
    if (same_assets && same_period)
        return cm
    end
    reordering = map(x -> findfirst(x .== cm.cm.labels)[1], labels)
    Acor = same_assets ? cm.cm.correlation[reordering, reordering] :
        Hermitian(cm.cm.correlation[reordering, reordering])
    Avol = same_period ? cm.cm.volatility[reordering] :
        convert_vol(
        cm.cm.volatility[reordering],
        cm.cm.time_period_per_unit,
        time_period_per_unit,
    )
    newmat = CovarianceMatrix(Acor, Avol, labels, time_period_per_unit)
    Adrift = same_period ? cm.means[reordering] : convert_drift(
        cm.drifts[reordering],
        cm.cm.time_period_per_unit,
        time_period_per_unit,
        )
    Amean = cm.means[reordering]
    return CovarianceModel(newmat, Amean, Adrift)
end

"""
    get_conditional_distribution(covar::CovarianceModel,
        conditioning_assets::Vector{Symbol},
        conditioning_asset_returns::Vector{<:Real},
        data_return_interval = covar.cm.time_period_per_unit)

Calculates the conditional gaussian distribution after conditioning on a few assets.
Note that the conditional mean is baked into the .mean attribute of the CovarianceModel. Thus this mean
is as of the end point of the return interval you are conditioning on.
### Takes
* `covar` - A `CovarianceMatrix`.
* `conditioning_assets` - A `Vector` of labels for the assets to condition on.
* `conditioning_asset_returns` - A `Vector` of the returns for each of these assets.
* `data_return_interval` - The time period the returns is for.
### Returns
* A `CovarianceModel`.
"""
function get_conditional_distribution(covar::CovarianceModel,
                                      conditioning_assets::Vector{Symbol},
                                      conditioning_asset_returns::Vector{<:Real},
                                      data_return_interval = covar.cm.time_period_per_unit)
    non_covariance_moves = [get_drift(covar, x, data_return_interval) + get_mean(covar, x) for x in conditioning_assets]
    resid_returns = conditioning_asset_returns .- non_covariance_moves

    covar_matrix = covariance(covar.cm, data_return_interval)
    non_conditioning_assets = setdiff(covar.cm.labels, conditioning_assets)
    asset_indices = [findfirst(asset .== covar.cm.labels) for asset in non_conditioning_assets]
    conditioning_indices = [findfirst(asset .== covar.cm.labels) for asset in conditioning_assets]
    # Segmenting the covariance matrix.
    sigma11 = covar_matrix[asset_indices,asset_indices]   
    sigma12 = covar_matrix[asset_indices,conditioning_indices]
    sigma21 = covar_matrix[conditioning_indices,asset_indices]
    sigma22 = covar_matrix[conditioning_indices,conditioning_indices]
    mu1 = zeros(length(asset_indices))
    mu2 = zeros(length(conditioning_indices))
    weights = sigma12 / sigma22
    conditional_mu = mu1 + weights * (resid_returns - mu2)
    conditional_sigma = sigma11 - weights * sigma21
    corr, vol = cov_to_cor_and_vol(conditional_sigma, data_return_interval, data_return_interval)

    new_covar = CovarianceMatrix(corr,vol, non_conditioning_assets,data_return_interval)

    # drift and means
    newdrift = [get_drift(covar,x, data_return_interval) for x in non_conditioning_assets] 
    newmean  = [get_mean(covar,x) for x in non_conditioning_assets] + conditional_mu

    return CovarianceModel(new_covar, newmean, newdrift)
end
