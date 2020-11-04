
at_times = get_all_refresh_times(ts, assets)
dd_compiled = latest_value(ts, at_times; assets = assets)
dd = get_returns(dd_compiled)
rets = dd[:,:BARC]

# This is going to be error prone. Because you invert Sigma. It is expensive, finicky (in that an optimisation is required).
# Also for high enough dimension we will get a negative determinant and it all goes to pieces.


a2 = 0.0001
sigma2 = 0.1
delta = spacing
x = [sigma2, a2]
function log_likelihood_AS_Fan_XIU(x, delta, Y)
    println(x)
    sigma2 = (x[1]); a2 = (x[2]) # So they are in the positive domain.
    n = length(Y)
    # Now for the det we have the problem if sigma2 * delta + 2a2 is far from 1 then the determinant goes to zero or infinity
    # if it is a big matrix. So to prevent that we use the trick det(kA) = k^n det(A). So log(det(kA)) = n log(k) + log(det(A)).
    # k = 1/(sigma2 delta + 2a2).
    # But then if n = 1000 or some high number the determinant will still get low as it is 1 - (n-1)offdiagonal^2 + ... in the case with ones on diagonal.
    k = 1/(sigma2 * delta + 2a2)
    Omega = SymTridiagonal(repeat([1.0], n), repeat([-a2 * k], n))
    invv = (1/k) .* (inv(Omega))
    logdett = n * log(k) + log(det(Omega))
    return -0.5 * logdett - 0.5 * (   transpose(Y) * invv * Y   )
end


using Optim
opt = optimize(x -> -log_likelihood_AS_Fan_XIU(x, delta, rets), [0.03, 0.001], BFGS())

log_likelihood_AS_Fan_XIU([0.03, 0.001], delta, rets)
log_likelihood_AS_Fan_XIU([0.3, 0.01], delta, rets)
log_likelihood_AS_Fan_XIU([3, 1], delta, rets)
log_likelihood_AS_Fan_XIU([3.463348865634988e7, 444098.02729438327], delta, rets)
