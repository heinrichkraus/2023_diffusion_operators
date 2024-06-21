abstract type AbstractLaplaceOperator <: AbstractDifferentialOperator end

Base.@kwdef struct Laplace_WLSQ{DD <: AbstractDiagonalDominance} <: AbstractLaplaceOperator
    order::Int  = 2
    dd::DD      = OneDimensionalCorrection_Default()
    scale::Bool = true
end

struct Laplace_Voronoi <: AbstractLaplaceOperator end

function laplace_row(i, pc::PointCloud, method::Laplace_WLSQ; kwargs...)
    order = method.order
    order < 2 && @error "Order of WLSQ-based Laplace operator is $(order) which is less than 2!"

    K, W = localmatrices(i, method.order, pc; scale=method.scale, kwargs...)
    dim = dimension(pc)
    rhs = zeros(nfuns(order, dim))

    if dim == 2
        rhs[4] = 2
        rhs[6] = 2
    else
        error("Dimension $dim not implemented.")
    end

    return  1 / pc.p.h[i]^2 * leastsquares(K, W, rhs, method.dd)

end

function laplace_row(i, pc, ::Laplace_Voronoi)
    surfsize, volume = voronoi(i, pc)
    neighbors = pc.p.neighbors[i]
    dij = norm.(pc.p.x[neighbors] .- (pc.p.x[i], ))

    c = zeros(precision(pc), length(neighbors))
    c[2:end] = @. surfsize[2:end] / (volume * dij[2:end])
    c[1]     = -sum(c[2:end])

    return c
end

function laplace(pc; method = Laplace_WLSQ())
    M = zero(pc.sparsity)
    for i = eachindex(pc)
        isboundary(pc.p[i]) && continue
        M[i, pc] = laplace_row(i, pc, method)
    end
    return M
end
