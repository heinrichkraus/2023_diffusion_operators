function neumann_weak(i, pc::PointCloud{2})
    edgelength, area = voronoi(i, pc)
    neighbors = pc.p.neighbors[i]
    eta = pc.p.eta[neighbors]

    v = zeros(precision(pc), length(neighbors))

    for (k, j) = enumerate(neighbors)
        edgelength[k] == 0.0 && continue
        # etaij = (eta[1] + eta[k]) / 2
        etaij = 2 * (eta[1] * eta[k]) / (eta[1] + eta[k])
        # etaij = sqrt(eta[1] * eta[k])
        W = etaij * ( edgelength[k] / norm(pc.p.x[i] - pc.p.x[j]) )
        v[1] -= W
        v[k] += W
    end

    return v / area
end