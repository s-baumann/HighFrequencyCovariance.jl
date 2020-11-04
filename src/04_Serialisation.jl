
import StochasticIntegrals.to_dataframe
"""
Convert a CovarianceMatrix to a dataframe format.
"""
function to_dataframe(covar::CovarianceMatrix, othercols::Dict = Dict{Symbol,Any}(); delete_duplicate_correlations::Bool = true)
    d = size(covar.correlation)[1]
    corrs = DataFrame(asset1 = vcat(map(a -> repeat([a], d), covar.labels  )...),asset2 = Array{Union{Symbol,Missing}}(repeat(covar.labels, d)),value = vec(covar.correlation))
    corrs[!,:variable] = repeat([:correlation], nrow(corrs))
    if delete_duplicate_correlations corrs = corrs[findall(map(a -> findfirst(covar.labels .== a), corrs[:,:asset1]) .< map(a -> findfirst(covar.labels .== a), corrs[:,:asset2])),:] end
    vols = DataFrame(asset1 = covar.labels, value = covar.volatility)
    vols[!,:variable] = repeat([:volatility], nrow(vols))
    vols[!,:asset2]   = repeat([missing], nrow(vols))
    result = append!(corrs, vols)
    for k in keys(othercols)
        result[!,k] = repeat([othercols[k]], nrow(result))
    end
    return result
end


"""
Convert a CovarianceMatrix to a dataframe format.
"""
function dataframe_to_covariancematrix(dd::DataFrame)
    vol_dd = dd[dd[:,:variable] .== :volatility,:]
    assets = vol_dd[:,:asset1]
    vols   = vol_dd[:,:value]
    assets = vol_dd[:,:asset1]

    cor_dd = dd[dd[:,:variable] .== :correlation,:]
    N = length(assets)
    mat = zeros(N,N)
    for i in 1:N
        asseti = assets[i]
        for j in i:N
            assetj = assets[j]
            if (i == j)
                mat[i,j] = 1
            else
                c1 = cor_dd[cor_dd[:,:asset1] .== asseti,:]
                c1 = c1[c1[:,:asset2] .== assetj,:]
                if nrow(c1) > 0
                    mat[i,j] = c1[1,:value]
                else
                    c2 = cor_dd[cor_dd[:,:asset2] .== asseti,:]
                    c2 = c1[c1[:,:asset1] .== assetj,:]
                    if nrow(c2) == 0 error("The correlation between ", asseti, " and ", assetj, " is not in this dataframe.") end
                    mat[i,j] = c2[1,:value]
                end
            end
        end
    end
    return CovarianceMatrix(Hermitian(mat), vols, assets)
end
