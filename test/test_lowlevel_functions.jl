using DataFrames
using LinearAlgebra
using Statistics: std, var, mean, cov
using HighFrequencyCovariance
using Random

ts = DataFrame(:lab => [:A,:B,:A,:A,:A,:B,:A,:B,:B], :tiempo => [1,2,3,4,5,5,6,7,8],
               :value => [1.01,2.0,1.011,1.02,1.011,2.2,1.0001,2.2,2.3])
tss = SortedDataFrame(ts, :tiempo, :lab, :value)
tss = SortedDataFrame(tss, :tiempo, :lab, :value) # Testing the constructor that takes a SortedDataFrame.
assets = get_assets(tss, 0)

function ~(A, B)
    abs(A - B) < 10*eps()
end

# Latest value test
latest_value(tss, [2]; assets = assets)[:,:A][1] ~ 1.01
latest_value(tss, [2]; assets = assets)[:,:B][1] ~ 2.0

# Subsetting test
nrow(subset_to_tick(tss, 4).df) == 4

# Subset to time
nrow(subset_to_time(tss, 5.5).df) == 6

# duration
duration(tss)  ~ 8-1

# ticks per asset
ticks_per_asset(tss, assets) == Dict([:A, :B] .=> [5, 4])
# Refresh times
refresh = get_all_refresh_times(tss, assets) == [2,5,7]

# Testing two scales covariance when there is not much data.
tsc = two_scales_covariance(tss, [:A, :B])
sum(isnan.(tsc.volatility)) == length(tsc.volatility)
