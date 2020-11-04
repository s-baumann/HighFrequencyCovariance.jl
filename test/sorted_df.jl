using DataFrames
using Random
twister = MersenneTwister(1)
df = DataFrame(:A => [1,2,1,2,1,2,1,2,3,3],
               :B => [:a,:b,:a,:b,:a,:b,:a,:b,:a,:b])
df.price = rand(twister,10)

sorting = [:B,:A]
sort(df, sorting)
