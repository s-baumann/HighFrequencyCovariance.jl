
function cluster_based_on_correlation_matrix(corr_matrix; linkage=:ward)
    # Convert to distance matrix
    dist_matrix = pairwise(Euclidean(), sqrt.(0.5 .* (1 .- corr_matrix)))
    # Hierarchical clustering
    hc = hclust(dist_matrix, linkage=linkage)
    return hc
end

function reorder_according_to_clustering(covar::CovarianceMatrix; linkage::Symbol = :ward)
    hc = cluster_based_on_correlation_matrix(covar.correlation; linkage=linkage)
    reordered_labels = covar.labels[hc.order]
    reordered_covar = rearrange(covar, reordered_labels) 
    return reordered_covar
end