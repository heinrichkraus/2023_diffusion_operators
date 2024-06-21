function directional_derivative_row(dir, point, h, neighbors, order, pc, dd)
    K, W = localmatrices(point, h, neighbors, order, pc; scale = true)

    dim  = dimension(pc)
    rhs  = zeros(nfuns(order, dim))

    if dim == 2
        rhs[2] = dir[1]
        rhs[3] = dir[2]
    else
        error("Dimension $dim not implemented")
    end

    return 1 / h * leastsquares(K, W, rhs, dd)
end

function gradient_row(point, h, neighbors, order, pc, dd)
    K, W = localmatrices(point, h, neighbors, order, pc; scale = true)

    dim  = dimension(pc)
    rhs  = zeros(nfuns(order, dim), dim)

    for i = 1:dim, j = 1:dim
        rhs[i + 1, j] = i == j ? 1 : 0
    end

    return 1 / h * transpose(leastsquares(K, W, rhs, dd))
end


function directional_derivative(dir, pc; dd = DD_Off(), order = 2)

    dir == zero(dir) && error("Cannot compute directional derivative without direction!")
    length(dir) != dimension(pc) && error("Dimension mismatch. Dimension of direction has to be the same as dimension of the point cloud.")
    order < 1 && error("Order needs to be at least 1 but it is $order.")

    M = zero(pc.sparsity)

    for i = eachindex(pc)
        M[i, pc] = directional_derivative_row(dir, pc.p.x[i], pc.p.h[i], pc.p.neighbors[i], order, pc, dd)
    end

    return M

end