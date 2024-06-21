weight_function(r) = exp(-5r)

function localmatrices(p, h, neighbors, order, pc;
    scale = false,
    weight_function = weight_function,
    kwargs...
)

    dim   = dimension(pc)
    ncols = length(neighbors)
    nrows = nfuns(order, dim)

    K = zeros(nrows, ncols)
    w = zeros(ncols)

    nrows > ncols && @warn "More conditions than neighbors!"

    k = 1
    if dim == 2
        for (k, j) = enumerate(neighbors)
            dp = pc.p.x[j] - p

            w[k] = weight_function(2 * norm(dp) / (h + pc.p.h[j]))

            if scale
                dp *= 1 / h
            end

            dx = dp[1]
            dy = dp[2]

            K[1, k] = 1

            if order > 0
                K[2, k] = dx
                K[3, k] = dy
            end

            if order > 1
                K[4, k] = dx^2
                K[5, k] = dx * dy
                K[6, k] = dy^2
            end

            if order > 2
                K[7, k] = dx^3
                K[8, k] = dx^2 * dy
                K[9, k] = dx * dy^2
                K[10, k] = dy^3
            end

            if order > 3
                K[11, k] = dx^4
                K[12, k] = dx^3 * dy
                K[13, k] = dx^2 * dy^2
                K[14, k] = dx * dy^3
                K[15, k] = dy^4
            end

        end
    else
        error("Dimension $dim not implemented!")
    end

    return K, Diagonal(w)
end

function localmatrices(i, order, pc; kwargs...)
    return localmatrices(pc.p.x[i], pc.p.h[i], pc.p.neighbors[i], order, pc; kwargs...)
end

nfuns(order, dim) = binomial(order + dim, dim)
