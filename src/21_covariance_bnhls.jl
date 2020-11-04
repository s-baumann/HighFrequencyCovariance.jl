################################################################################
#####                   BNHLS Covariance Estimation                        #####
################################################################################

"""
A kernel used in the bnhls covariance method.
"""
struct HFC_Kernel{R<:Real}
    f::Function
    abs_k_pp_0::R
    k_00::R
    c_star::R
    relative_asymptotic_efficiency::R
    domain::Tuple{R,R}
    function HFC_Kernel(f::Function, abs_k_pp_0::R, k_00::R, c_star::R, relative_asymptotic_efficiency::R, domain::Tuple{R,R}= (-Inf, Inf)) where R<:Real
        return new{R}(f, abs_k_pp_0, k_00, c_star, relative_asymptotic_efficiency, domain)
    end
end

function parzen_func(x::Real)
    if x < 0.5 return 1 -6*(x^2) + 6*(x^3) end
    return x < 1 ? 2*((1-x)^2) : 0.0
end
"""
A parzen kernel used in the bnhls covariance method.
"""
const parzen = HFC_Kernel(parzen_func, 12.0, 0.269, 3.51, 0.97, (-1.0,1.0))
fqs(x)   = abs(x) < 0.001 ? 1.0 : (3/(x^2))*(sin(x)/x - cos(x))
"""
A quadratic_spectral kernel used in the bnhls covariance method.
"""
const quadratic_spectral = HFC_Kernel(fqs, 0.2, 3*pi/5, 0.46, 0.93)
f_fejer(x)   = abs(x) < 0.001 ? 1.0 : (sin(x)/x)^2
"""
A fejer kernel used in the bnhls covariance method.
"""
const fejer  = HFC_Kernel(f_fejer, 2/3, pi/3, 0.84, 0.94)
f_tukey(x)   = (sin(exp(-x)*(pi/2)))^2
"""
A tukey_hanning kernel used in the bnhls covariance method.
"""
const tukey_hanning  = HFC_Kernel(f_tukey, (pi^2)/2, 0.52, 2.16, 1.06)
f_bnhls(x) = (1+x)*exp(-x)
"""
A bnhls_2008 kernel used in the bnhls covariance method.
"""
const bnhls_2008 = HFC_Kernel(f_bnhls, 1.0, 5/4, 0.96, 1.09)

"""
Realised autocovariance.
"""
function realised_autocovariance(returns::Array{R,2}, h::Integer) where R<:Real
    d = size(returns)[2]
    h = abs(h)
    start = h+1
    end_ = size(returns)[1]
    summed = zeros(d,d)
    for j in start:end_
        summed += returns[j,:] * transpose(returns[j-h,:])
    end
    return summed
end

function realised_autocovariance2(returns::Array{R,2}, h::Integer) where R<:Real
    d = size(returns)[2]
    h = abs(h)
    start = h+1
    end_ = size(returns)[1]
    summed = zeros(d,d)
    for j in start:end_
        summed += AtB_plus_BtA(returns[j,:], returns[j-h,:])
    end
    return summed
end


# Defining the following we have
# J = returns[j,:]
# M = returns[j-h,:]
# H = returns[j+h,:]
# Now in cases where for J we have a j-H and a j+H, we need to add up
# Jt(M) + Ht(J) + t(Jt(M) + Ht(J))
# Which simplifies down to
# X = H + M
# Jt(X) + Xt(J)
# t(Xt(J)) + Xt(J)
# In cases where we only have a minus (where we are above end1 in below notation) we want
# Jt(M) + t(Jt(M))
function realised_autocovariance_plus_its_transpose(returns::Array{R,2}, h::Integer) where R<:Real
    d = size(returns)[2]
    h = abs(h)
    start0 = h+1
    end0 = size(returns)[1]
    end1 = end0 - h
    gap = end1 - start0
    summed = Hermitian(zeros(d,d))
    intervals = Int.(ceil((end1 - start0)/h))
    twosideds = reduce(vcat, map(i -> collect( (start0 + i*h):(start0 + (i+1)*h-1)  )   , 0:2:(intervals-1)))
    l1 = length(twosideds)
    twosideds = twosideds[twosideds .< end1+1]
    l2 = length(twosideds)
    for j in twosideds
        X       = returns[j-h,:] + returns[j+h,:]
        summed += AtB_plus_BtA(X, returns[j,:])
    end
    #onesideds = end1:(end0-gap+1)
    #if gap < 0 onesideds = start0:end0 end
    onesideds = (maximum(twosideds)+h+1):end0
    for j in onesideds
        summed += AtB_plus_BtA(returns[j,:], returns[j-h,:])
    end
    onesideds = (maximum(twosideds)+1):min((minimum(twosideds)+h-1), end0)
    for j in onesideds
        summed += AtB_plus_BtA(returns[j,:], returns[j-h,:])
    end
    onesideds = (maximum(twosideds)+1):(maximum(twosideds) + l1 - l2)
    for j in onesideds
        summed += AtB_plus_BtA(returns[j,:], returns[j-h,:])
    end
    return summed
end

"""
This averages the first few and last few returns. We do this to returns rather than
prices (as suggested in BNHLS 2011).
### Takes
* returns - An array of returns
* m - Number of rows to average on each end.
### Returns
* An array of returns with ends averaged.
# References
Barndorff-Nielsen, O., Hansen, P.R., Lunde, A., Shephard, N. 2011. - Section 2.2.
"""
function preaveraging_end_returns(returns::Array{R,2}, m::Integer) where R<:Real
    N = size(returns)[1]
    if m >= (N/2) error("m is too high compared to the dimension of the returns matrix. Not possible to average away this much data.") end
    # Preaveraging of first few returns.
    first_m_rows  = returns[1:m,:]
    for i in 1:m first_m_rows[i,:] = first_m_rows[i,:] .* (-i/m) end
    first_row = sum(first_m_rows, dims = 1)
    # Preaveraging of last few returns.
    last_m_rows   = returns[(N-m+1):N,:]
    for i in 1:m last_m_rows[i,:] = last_m_rows[i,:] .* (i/m) end
    last_row = sum(last_m_rows, dims = 1)
    # returning the completed returns.
    new_returns   = deepcopy(returns[m:(N-m+1),:])
    new_returns[1,:]          = first_row
    new_returns[(N-2*m+2),:]  = last_row
    return new_returns
end

"""
This calculates covariance with the Multivariate realised kernel of BNHLS(2011). Given an arry of returns.
"""
function bnhls_covariance_estimate_given_returns(returns::Array{R,2}; kernel::HFC_Kernel{T}, H::Real, m::Integer) where R<:Real where T<:Real
    returns_end_averaged  = preaveraging_end_returns(returns, m)
    N = size(returns)[1]
    termination = Integer(floor(min(N, maximum(abs.(kernel.domain))*H)))
    summed = realised_autocovariance(returns,0) # as kernel.f(0) is always 1.
    for h in 1:termination
        kern = kernel.f(h/H)
        a1 = realised_autocovariance(returns,h)
        summed  += kern .* (a1 + transpose(a1)) # We are doing the negative h values at the same time as positives here.
        #autocov  = realised_autocovariance_plus_its_transpose(returns,h)
        #summed  += kern .* autocov # We are doing the negative h values at the same time as positives here.
        #a1 = realised_autocovariance2(returns,h)
        #summed  += kern .* a1
    end
    return Hermitian(summed)
end

"""
This calculates covariance with the Multivariate realised kernel oof BNHLS(2011)
### Takes
* ts - A SortedDataFrame containing the asset prices.
* assets - A vector with the asset names that we are interested in.
* regularisation - A regularisation function (or missing if not regularisation is to be done).
* only_regulise_if_not_PSD - Should regularisation only be done if the matrix is not PSD.
* kernel - A HFC_Kernel object. Parzen by default.
* H - The level of autocovariance to be estimated. By default this is kernel.c_star * N^0.6 where N is the average number of asset prices. (following the suggestion of BNHLS pag 154 but ignoring the noise term for simplicity)
* m - The number of start/end periods to average. This is 2 by default as recommened by BNHLS
* return_calc - The function used to calculate returns.
### Returns
* A CovarianceMatrix.
# References
Barndorff-Nielsen, O., Hansen, P.R., Lunde, A., Shephard, N. 2011. - The whole paper but particularly 2.2, 2.3 here. Kernels are in table 1. choices of H are discussed in section 3.4 of the paper.
"""
function bnhls_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = eigenvalue_clean,
                          only_regulise_if_not_PSD::Bool = false, kernel::HFC_Kernel{<:Real} = parzen, H::Real = kernel.c_star * ( mean(map(a -> length(ts.groupingrows[a]), assets))   )^0.6,
                          m::Integer = 2, return_calc::Function = simple_differencing)
    at_times = get_all_refresh_times(ts, assets)
    dd_compiled = latest_value(ts, at_times; assets = assets)
    dd = get_returns(dd_compiled; return_calc = return_calc)

    if m >= nrow(dd)/2
       println("Cannot estimate the correlation matrix with ", nrow(ts.df), " ticks as some variances are negative.")
       return make_nan_covariance_matrix(assets)
   end

    returns = Matrix(dd[:, assets])
    covariance = bnhls_covariance_estimate_given_returns(returns; kernel = kernel, H = H, m = m)

    # Regularisation
    dont_regulise = ismissing(regularisation) || (only_regulise_if_not_PSD && (minimum(eigen(covariance)[1]) < 0))
    covariance = dont_regulise ? covariance : regularisation(covariance, ts)

    # In some cases we get negative terms on the diagonal with this algorithm.
    negative_diagonals = findall(diag(covariance) .< eps())
    covar = make_nan_covariance_matrix(assets)
    if length(negative_diagonals) == 0
        cor, vols = cov2cor_and_vol(covariance, duration(ts))
        covar.correlation = cor
        covar.volatility = vols
     end
    return covar
end
