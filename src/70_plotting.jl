function plot_heatmap(dd::DataFrame, value_col::Symbol, rounded_value_column::Symbol, xcol::Symbol, ycol::Symbol;
                      xorder::Vector = sort(unique(dd[:, xcol])), yorder::Vector = sort(unique(dd[:, xcol])),
                      domainMin::Real = minimum(dd[:,value_col]), domainMid::Real = median(dd[:,value_col]), domainMax::Real = maximum(dd[:,value_col]),
                      title = "", width = 1000, height = 1000)
    p1= dd |> VegaLite.@vlplot(
        title = title,
        :rect,
        x={field=xcol, type="ordinal", sort=xorder},
        y={field=ycol, type="ordinal", sort=yorder},
        color={field=value_col, type="quantitative", scale={range = "diverging", domainMin=domainMin, domainMid=domainMid, domainMax=domainMax}},
        width=width,
        height=height
    )
    p2 = dd |> VegaLite.@vlplot(
        title = title,
        :text,
        text = {field=rounded_value_column},
        x={field=xcol, type="ordinal", sort=xorder},
        y={field=ycol, type="ordinal", sort=yorder},
        width=width,
        height=height
    )
    ptotal = @vlplot(background = "#fff") + p1 + p2
    return ptotal
end

function plot(covar::Union{CovarianceMatrix,CovarianceMatrix{R}},
                          covar2::Union{CovarianceMatrix,CovarianceMatrix{R}},
                          names_to_include = covar.labels; title = "", width = 1000, height = 1000) where R<:Real
    c12 = rearrange(covar, names_to_include)
    dd1 = DataFrame(Matrix(c12.correlation), c12.labels)
    # Putting the second covar matrix correlations in the top right.
    for i in 1:nrow(dd1)
        for j in (i+1):(ncol(dd1)-1)
            asset_i = c12.labels[i]
            asset_j = c12.labels[j]
            dd1[i,j] = HighFrequencyCovarianceTwo.get_correlation(covar2, asset_i, asset_j)
        end
    end
    dd1[!, :asset2] = c12.labels
    #
    long_cor = stack(dd1, c12.labels, variable_name = :asset1)
    ordering = string.(c12.labels)
    long_cor[!, :rounded_value] = string.(Int.(floor.(round.(long_cor[:, :value], digits = 2) * 100))) .* "%"
    p = plot_heatmap(long_cor, :value, :rounded_value, :asset1, :asset2;
        xorder = ordering, yorder = ordering,
        domainMin = -1.0, domainMid = 0.0, domainMax = 1.0,
        title = title, width = width, height = height)
    return p
end

function plot(covar::Union{CovarianceMatrix,CovarianceMatrix{R}},
                          names_to_include = covar.labels; title = "", width = 1000, height = 1000) where R<:Real
    c2 = rearrange(covar, names_to_include)
    dd = DataFrame(Matrix(c2.correlation), c2.labels)
    dd[!, :asset2] = c2.labels
    long_cor = stack(dd, c2.labels, variable_name = :asset1)
    ordering = string.(c2.labels)
    long_cor[!, :rounded_value] = string.(Int.(floor.(round.(long_cor[:, :value], digits = 2) * 100))) .* "%"
    p = plot_heatmap(long_cor, :value, :rounded_value, :asset1, :asset2;
        xorder = ordering, yorder = ordering,
        domainMin = -1.0, domainMid = 0.0, domainMax = 1.0,
        title = title, width = width, height = height)
    return p
end


function plot_dendrogram(covar::CovarianceMatrix; linkage::Symbol = :ward)
    hc = cluster_based_on_correlation_matrix(covar.correlation; linkage=linkage)
    return plot_dendrogram(hc; labels=covar.labels)
end

function plot_dendrogram(hc::Clustering.Hclust; labels=nothing)
    n = length(hc.order)
    coords = zeros(n + length(hc.height), 2)  # x, y coords for each node
    lines = []

    # Set leaf positions (x coordinate = order in dendrogram)
    for (i, idx) in enumerate(hc.order)
        coords[idx, :] = [i, 0.0]
    end

    # Internal nodes (merges)
    for i in 1:length(hc.height)
        left, right = hc.merge[i,:]
        left_idx  = left  < 0 ? -left  : n + left
        right_idx = right < 0 ? -right : n + right

        # Current node index
        current_idx = n + i
        height = hc.height[i]

        # Position this node in the middle of its children
        x = (coords[left_idx, 1] + coords[right_idx, 1]) / 2
        coords[current_idx, :] = [x, height]

        # Draw lines: verticals for children and horizontal connector
        push!(lines, scatter(x=[coords[left_idx,1], coords[left_idx,1]], y=[coords[left_idx,2], height],
                                mode="lines", line=attr(color="black")))
        push!(lines, scatter(x=[coords[right_idx,1], coords[right_idx,1]], y=[coords[right_idx,2], height],
                                mode="lines", line=attr(color="black")))
        push!(lines, scatter(x=[coords[left_idx,1], coords[right_idx,1]], y=[height, height],
                                mode="lines", line=attr(color="black")))
    end

    # Prepare x-tick labels (if provided)
    if isnothing(labels)
        labels = string.(1:n)  # Default labels as "1", "2", ..., "n"
    end
    if length(labels) != n
        error("Number of labels must match number of data points")
    end
    # Order them by hc.order (the dendrogram leaf order)
    tickvals = 1:n
    ticktext = labels[hc.order]
    xticks = attr(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    # Creating the layout
    layout = Layout(
        title="Dendrogram",
        xaxis=merge(attr(showticklabels=true, zeroline=false), xticks),
        yaxis=attr(title="Height"),
        showlegend=false
    )

    return PlotlyJS.plot(Vector{GenericTrace}(lines), layout)
end
