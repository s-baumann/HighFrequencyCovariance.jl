using Gadfly
using DataFrames
using Statistics: std, var, mean, cov, median
using CSV
using Glob
using Cairo
using Fontconfig
plot_folder = "C:/Dropbox/Stuart/Papers/high_frequency_covariance/plots/"

fldr = "C:/Dropbox/Stuart/Papers/high_frequency_covariance/convergences/"
files = glob("*.csv",fldr)

dd = CSV.read(files[1])
for i in 2:length(files)
    global dd = vcat(dd, CSV.read(files[i]))
end

how_many_nans(x) = mean(isnan.(x))
function mean_ex_nans(x)
     vals = x[findall(isnan.(x) .== false)]
     return length(vals) > 0 ? median(vals) : NaN
end
dd[!, :abserror] = abs.(dd[:,:value] .- dd[:,:true_value])

dd2 = combine(groupby(dd, [:dimensions, :with_noise, :syncronous, :number_of_paths, :method, :pathnum, :variable]), :abserror => mean_ex_nans)
rename!(dd2, :abserror_mean_ex_nans  => :MAE)

aa = combine(groupby(dd2, [:dimensions, :with_noise, :syncronous, :number_of_paths, :method, :variable]), :MAE => mean_ex_nans, :MAE => how_many_nans)





aa = sort(aa, :dimensions)

aa[!,:dims] = map(x -> string(x, " assets"), aa[:,:dimensions])
aa[!,:ticks_per_asset] = aa[:,:number_of_paths] ./ aa[:,:dimensions]
aa[!,:estimation] = map(i -> string( Bool(aa[i,:syncronous]) ? "Syncronous updates" : "Asyncronous updates", "\n" ,  Bool(aa[i,:with_noise]) ? "with " : "without ", " noise"    ), 1:nrow(aa) )



#aa = aa[aa.method .!= "spectral_covariance",:]


yvar = :MAE_mean_ex_nans
bb = aa[aa.variable .== "correlation",:]
plt = plot(bb, xgroup=:dims, ygroup=:estimation, Geom.subplot_grid(layer(x=:ticks_per_asset,y = yvar, color=:method, Geom.point),
           layer(x=:ticks_per_asset,y = yvar, color=:method, Geom.line)),
           Scale.x_log10, Scale.y_log10, Guide.xlabel("Average number of updates per asset"), Guide.ylabel("Mean Absolute Error in estimated correlations"), style(key_position = :bottom),
           Guide.ColorKey(title = ""), Guide.Title("Accuracy in estimating correlation matrices"))
img = PDF(string(plot_folder, "correlation_imprecision.pdf"), 30cm, 30cm)
draw(img, plt)




yvar = :MAE_mean_ex_nans
bb = aa[aa.variable .== "volatility",:]
plt = plot(bb, xgroup=:dims, ygroup=:estimation, Geom.subplot_grid(layer(x=:ticks_per_asset,y = yvar, color=:method, Geom.point),
           layer(x=:ticks_per_asset,y = yvar, color=:method, Geom.line)),
           Scale.x_log10, Scale.y_log10, Guide.xlabel("Average number of updates per asset"), Guide.ylabel("Mean Absolute Error in estimated volatilities"), style(key_position = :bottom),
           Guide.ColorKey(""), Guide.Title("Accuracy in estimating volatilities"))
img = PDF(string(plot_folder, "volatility.pdf"), 30cm, 30cm)
draw(img, plt)

#########################################
# Time and space complexity
#############################


fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/time_and_space_complexity_up_dimensions.csv")
aa = CSV.read(fname)
aa[:,:Bytes_mean] .= (aa[:,:Bytes_mean] ./ (1024*1024))
aa = melt(aa, [:Method, :Dims, :ticks_per_dim])
bb = aa[aa[:,:ticks_per_dim] .== maximum(aa[:,:ticks_per_dim]),  :]
bb[!,:var] .=  "Seconds to estimate"
bb[bb[:,:variable] .== "Bytes_mean",:var] .=  "Memory allocated (MiB)"
bb = bb[bb[:,:var] .!= "Memory allocated (MiB)",:]


plt = plot(bb, ygroup=:var, Geom.subplot_grid(layer( x = :Dims , y = :value, color=:Method, Geom.point),
           layer( x = :Dims , y = :value, color=:Method, Geom.line), free_y_axis =true),
           Scale.x_log10, Scale.y_log10, Guide.xlabel("Number of Assets"), Guide.ylabel(""), style(key_position = :bottom),
           Guide.ColorKey(title = ""), Guide.Title(" "))
img = PDF(string(plot_folder, "complexity-dimensions.pdf"), 15cm, 15cm)
draw(img, plt)



fname = string("C:/Dropbox/Stuart/Papers/high_frequency_covariance/time_and_space_complexity_up_obs.csv")
aa = CSV.read(fname)
aa[:,:Bytes_mean] .= (aa[:,:Bytes_mean] ./ (1024*1024))
aa = melt(aa, [:Method, :Dims, :ticks_per_dim])
bb = aa[aa[:,:Dims] .== maximum(aa[:,:Dims]),  :]
bb[!,:var] .=  "Seconds to estimate"
bb[bb[:,:variable] .== "Bytes_mean",:var] .=  "Memory allocated (MiB)"
bb = bb[bb[:,:var] .!= "Memory allocated (MiB)",:]

plt = plot(bb, ygroup=:var, Geom.subplot_grid(layer( x = :ticks_per_dim , y = :value, color=:Method, Geom.point),
           layer( x = :ticks_per_dim , y = :value, color=:Method, Geom.line), free_y_axis =true),
           Scale.x_log10, Scale.y_log10, Guide.xlabel("Average price updates per asset"), Guide.ylabel(""), style(key_position = :bottom),
           Guide.ColorKey(title = ""), Guide.Title(" "))
img = PDF(string(plot_folder, "complexity-obs.pdf"), 15cm, 15cm)
draw(img, plt)
