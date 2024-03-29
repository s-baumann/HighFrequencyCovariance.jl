"""
     orthogonal_sine(t::Real, j::Real, block_num::Integer, block_width::Real)
This is the orthogonal sine funcion from the paper.
### References
Bibinger M, Hautsch N, Malec P, Reiss M (2014). “Estimating the quadratic covariation matrix from noisy observations: Local method of moments and efficiency.” The Annals of Statistics, 42(4), 1312–1346. doi:10.1214/14-AOS1224.
"""
@inline function orthogonal_sine(t::Real, j::Real, block_num::Integer, block_width::Real)
    # Note for simplicity and speed I am going to assume that we will never call this after (k+1)h_n. This is due to the way it will be calculated in the loop.
    start_of_block = block_num * block_width
    return t > start_of_block ?
           ((sqrt(2 * block_width)) / (j * pi)) *
           sin((j * pi * (t - start_of_block)) / block_width) :
           0.0
end


"""
    spectral_lmm_array(
        ts::SortedDataFrame,
        assets::Vector{Symbol} = get_assets(ts);
        regularisation::Union{Missing,Symbol} = :covariance_default,
        regularisation_params::Dict = Dict(),
        only_regulise_if_not_PSD::Bool = false,
        numJ::Integer = 100,
        num_blocks::Integer = 10,
        block_width::Real = (maximum(ts.df[:, ts.time])) / num_blocks,
        microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2],
    )

Estimation of a CovarianceMatrix using the spectral covariance method.
### Inputs
* `ts` - The tick data.
* `assets` - The assets you want to estimate volatilities for.
* `regularisation` - A symbol representing what regularisation technique should be used. If missing no regularisation is performed.
* `regularisation_params` - keyword arguments to be consumed in the regularisation algorithm.
* `only_regulise_if_not_PSD` - Should regularisation only be attempted if the matrix is not psd already.
* `numJ` - The number of J values. See the paper for details.
* `num_blocks` - The number of blocks to split the time frame into. See the paper for details.
* `block_width` - The width of each block to split the time frame into.
* `microstructure_noise_var` - Estimates of microstructure noise variance for each asset.
### Returns
* A `CovarianceMatrix`.

### References
Bibinger M, Hautsch N, Malec P, Reiss M (2014). “Estimating the quadratic covariation matrix from noisy observations: Local method of moments and efficiency.” The Annals of Statistics, 42(4), 1312–1346. doi:10.1214/14-AOS1224.
"""

function spectral_lmm_array(
    ts::SortedDataFrame,
    assets::Vector{Symbol} = get_assets(ts);
    regularisation::Union{Missing,Symbol} = :covariance_default,
    regularisation_params::Dict = Dict(),
    only_regulise_if_not_PSD::Bool = false,
    numJ::Integer = 100,
    num_blocks::Integer = 10,
    block_width::Real = (maximum(ts.df[:, ts.time])) / num_blocks,
    microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2],
)
    numAssets = length(assets)
    SS = zeros(num_blocks, numJ, numAssets)
    indices = Dict(assets .=> collect(1:numAssets))
    lasttimes = zeros(numAssets)
    lastvalues = zeros(numAssets)
    H_kn = zeros(num_blocks, numAssets)
    for i = 1:nrow(ts.df)
        asset = ts.df[i, ts.grouping]
        if (asset in assets) == false
            continue
        end
        assetslot = indices[asset]
        newtime = ts.df[i, ts.time]
        newval = ts.df[i, ts.value]
        if lasttimes[assetslot] <= 0
            lasttimes[assetslot] = newtime
            lastvalues[assetslot] = newval
            continue
        end
        timegap = (newtime - lasttimes[assetslot])
        centertime = lasttimes[assetslot] + 0.5 * timegap
        ret = simple_differencing([newval], [lastvalues[assetslot]])[1]
        lasttimes[assetslot] = newtime
        lastvalues[assetslot] = newval

        block_num = min(num_blocks - 1, Int(floor(newtime / block_width)))

        # Looking over j values for the S Array.
        for j = 1:numJ
            SS[block_num+1, j, assetslot] +=
                (pi * j / block_width) *
                ret *
                orthogonal_sine(centertime, j, block_num, block_width)
        end
        # Now doing the H_k^n matrix which is independent of j.
        H_kn[block_num+1, assetslot] +=
            (timegap^2) * (microstructure_noise_var[asset] / block_width)
    end
    # Now we assemble the covariance matrices for every j,k.
    R = eltype(SS)
    uncorrected_covar_matrices = Array{Array{Hermitian{R},1},1}(undef, num_blocks)
    for block_num = 1:num_blocks
        hermitian_array = Array{Hermitian{R},1}(undef, numJ)
        for j = 1:numJ
            hermitian_array[j] =
                Hermitian(SS[block_num, j, :] * transpose(SS[block_num, j, :]))
        end
        uncorrected_covar_matrices[block_num] = hermitian_array
    end

    # The efficient weights (see page 1324 of Bibinger, Hautsch, Malec & Reiss 2014) are not implemented here. It is highly error proone as
    # many inversions are required which is 1) computationally intensive and 2) can cause singularity errors as there is no guarantee of psd.
    # So going to fall back to equal weighting.

    corrected_matrices = Array{CovarianceMatrix{R},1}(undef, num_blocks)
    for block_num = 1:num_blocks
        block_H = Diagonal(H_kn[block_num, :])
        adjustments = ((pi .* collect(1:numJ)) ./ block_width) .^ 2
        block_adjustments = Array{Diagonal,1}(undef, numJ)
        for nn = 1:numJ
            block_adjustments[nn] = adjustments[nn] * block_H
        end
        corrected_mats = uncorrected_covar_matrices[block_num] .- block_adjustments
        mat = Hermitian(sum(corrected_mats .* (1 / length(corrected_mats))))

        # Regularisation
        dont_regulise =
            ismissing(regularisation) || (only_regulise_if_not_PSD && is_psd_matrix(mat))
        mat = dont_regulise ? mat :
            regularise(mat, ts, assets, regularisation; regularisation_params...)

        # In some cases we get negative terms on the diagonal with this algorithm.
        non_positive_diagonals = findall(diag(mat) .<= 0)
        covar = make_nan_covariance_matrix(assets, ts.time_period_per_unit)
        if length(non_positive_diagonals) < 1
            corr, vols = cov_to_cor_and_vol(mat, 1) # No adjustment for duration as we already have the spot matrix.
            copy!(covar.correlation, corr)
            copy!(covar.volatility, vols)
        end
        corrected_matrices[block_num] = covar
    end
    return corrected_matrices
end

"""
    spectral_covariance(
        ts::SortedDataFrame,
        assets::Vector{Symbol} = get_assets(ts);
        regularisation::Union{Missing,Symbol} = :covariance_default,
        regularisation_params::Dict = Dict(),
        only_regulise_if_not_PSD::Bool = false,
        numJ::Integer = 100,
        num_blocks::Integer = 10,
        block_width::Real = (maximum(ts.df[:, ts.time]) - minimum(ts.df[:, ts.time])) /
                            num_blocks,
        microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2],
    )

Estimation of a CovarianceMatrix using the spectral covariance method.
### Inputs
* `ts` - The tick data.
* `assets` - The assets you want to estimate volatilities for.
* `regularisation` - A symbol representing what regularisation technique should be used. If missing no regularisation is performed.
* `regularisation_params` - keyword arguments to be consumed in the regularisation algorithm.
* `only_regulise_if_not_PSD` - Should regularisation only be attempted if the matrix is not psd already.
* `numJ` - The number of J values. See the paper for details.
* `num_blocks` - The number of blocks to split the time frame into. See the paper for details.
* `block_width` - The width of each block to split the time frame into.
* `microstructure_noise_var` - Estimates of microstructure noise variance for each asset.
### Returns
* A `CovarianceMatrix`.

### References
Bibinger M, Hautsch N, Malec P, Reiss M (2014). “Estimating the quadratic covariation matrix from noisy observations: Local method of moments and efficiency.” The Annals of Statistics, 42(4), 1312–1346. doi:10.1214/14-AOS1224.
"""
function spectral_covariance(
    ts::SortedDataFrame,
    assets::Vector{Symbol} = get_assets(ts);
    regularisation::Union{Missing,Symbol} = :covariance_default,
    regularisation_params::Dict = Dict(),
    only_regulise_if_not_PSD::Bool = false,
    numJ::Integer = 100,
    num_blocks::Integer = 10,
    block_width::Real = (maximum(ts.df[:, ts.time]) - minimum(ts.df[:, ts.time])) /
                        num_blocks,
    microstructure_noise_var::Dict{Symbol,<:Real} = two_scales_volatility(ts, assets)[2],
)
    corrected_matrices = spectral_lmm_array(
        ts,
        assets;
        regularisation = regularisation,
        regularisation_params = regularisation_params,
        only_regulise_if_not_PSD = only_regulise_if_not_PSD,
        numJ = numJ,
        num_blocks = num_blocks,
        block_width = block_width,
        microstructure_noise_var = microstructure_noise_var,
    )
    weights = repeat([1.0], length(corrected_matrices))
    spectral_estimate = combine_covariance_matrices(corrected_matrices, weights, weights)
    return spectral_estimate
end
