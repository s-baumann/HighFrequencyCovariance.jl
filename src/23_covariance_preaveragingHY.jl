function get_preaveraged_prices(ts::SortedDataFrame, asset::Symbol, k_n::Real, gs::Vector, return_calc::Function)
   prices = Array(ts.df[ts.groupingrows[asset],ts.value])
   times = Array(ts.df[ts.groupingrows[asset],ts.time])
   durations = Array(times[2:end] .- times[1:(end-1)])
   times = times[2:end]
   diffs = return_calc(Array(prices[2:end]), Array(prices[1:(end-1)]), durations, asset)
   lastrow = length(diffs) - k_n + 1
   if lastrow < 1 return missing, missing end
   preaveraged_returns = Array{eltype(diffs),1}(undef, lastrow)
   for i in 1:lastrow
       preaveraged_returns[i] = sum(diffs[i:(i+k_n-2)] .* gs)
   end
   return preaveraged_returns, times
end

function HY_n(A::Tuple{Vector{<:Real},Vector{<:Real}}, B::Tuple{Vector{<:Real},Vector{<:Real}}, k_n::Integer, psi::Real)
   # See top of page 62 of Christensen et al.
   cov_sum = 0.0
   for i in 1:(length(A[1]) - k_n + 2)
      Y_i = A[1][i]
      i_start = A[2][i]
      i_end = A[2][i+k_n]
      j_index_i_start = max(1, searchsortedfirst(B[2], i_start) - k_n) # Because an observation k_n ticks ago for the other asset also overlaps.
      j_index_i_end   = min(length(B[1]),searchsortedlast(B[2], i_end))
      for j in j_index_i_start:j_index_i_end
          cov_sum += Y_i * B[1][j]
      end
   end
   return cov_sum / ((psi * k_n)^2)
end

# This is the 3rd equation in the aymptotic theory section of the paper. We don't use the above version because sometimes that gives us negative variances.
univariate_HYn(vect::Vector, k_n::Real, psi::Real) = sum(map(i -> vect[i] * sum( map(j -> vect[i-j], (1-k_n):(k_n-1)  ) ), k_n:(length(vect)-2k_n+1)))/((psi*k_n)^2)


g = (f = x-> min(x, 1-x), psi = 0.25)

"""
Estimation of the CovarianceMatrix using preaveraging method.
Christensen K, Podolskij M, Vetter M (2013). “On covariation estimation for multivariate continuous Itô semimartingales with noise in non-synchronous observation schemes.” Journal of Multivariate Analysis, 120, 59–84. doi:10.1016/j.jmva.2013.05.002.
"""
function preaveraged_covariance(ts::SortedDataFrame, assets::Vector{Symbol} = get_assets(ts); regularisation::Union{Missing,Function} = eigenvalue_clean,
                             only_regulise_if_not_PSD::Bool = false, theta::Real = 0.15, g::NamedTuple = g, return_calc::Function = simple_differencing)
   # The defaults are from the paper (Christensen et al 2013). theta and the formula for k_n is from halfway down page 67. g is from page 64.
   number_of_ticks = nrow(ts.df)
   k_n = Int(ceil(min(number_of_ticks/2,theta * sqrt(number_of_ticks))))
   gs = g.f.( collect(1:1:(k_n-1)) ./ k_n )
   prev_prices = get_preaveraged_prices.(Ref(ts), assets, k_n, Ref(gs), Ref(return_calc))

  lens = findall(map(i -> ismissing(prev_prices[i][1]), 1:length(prev_prices)))
  if length(lens) > 0
     @warn string("Cannot estimate the correlation matrix with ", number_of_ticks, " ticks. There are insufficient ticks for ", assets[lens])
     return make_nan_covariance_matrix(assets)
  end

   N = length(assets)
   # First doing HY_n from equation 2.5
   HYn = zeros(N,N)
   for i in 1:N
      A = prev_prices[i]
      for j in i:N
         if i ==j
            HYn[i,i] = univariate_HYn(A[1], k_n, g.psi)
         else
            B = prev_prices[j]
            HYn[i,j] = HY_n(A, B, k_n, g.psi)
         end
      end
   end
   HYn = Hermitian(HYn)
   # Regularisation
   dont_regulise = ismissing(regularisation) || (only_regulise_if_not_PSD && is_psd_matrix(HYn))
   HYn = dont_regulise ? HYn : regularisation(HYn, ts, assets)

   # In some cases we get negative terms on the diagonal with this algorithm.
   negative_diagonals = findall(diag(HYn) .< eps())
   covar = make_nan_covariance_matrix(assets)
   if length(negative_diagonals) == 0
       corr, _ = cov2cor_and_vol(HYn, 1)
       covar.correlation = corr
    end
   # We can use this to get the correlation matrix but the variances are too low - as a result of preveraging.
   # We will instead use the two scales vol of Zhang, mykland, Ait-Sahalia 2005.
   voldict = two_scales_volatility(ts, assets; return_calc = return_calc)[1]
   vols = map(a -> voldict[a], assets)
   covar.volatility = vols
   return covar
end
