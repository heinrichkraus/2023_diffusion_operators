function approximation_row(point, h, neighbors, order, pc, dd)
    K, W = localmatrices(point, h, neighbors, order, pc; scale = true)

    dim  = dimension(pc)
    rhs  = zeros(nfuns(order, dim))

    if dim == 2
        rhs[1] = one(precision(pc))
    else
        error("Dimension $dim not implemented")
    end

    return leastsquares(K, W, rhs, dd)
end

function approximation(pc; order = 2, dd = DD_Off())
    M = zero(pc.sparsity)
    for i = eachindex(pc)
        M[i, pc] = approximation_row(pc.p.x[i], pc.p.h[i], pc.p.neighbors[i], order, pc, dd)
    end
    return M
end