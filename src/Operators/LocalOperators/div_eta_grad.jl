abstract type AbstractDiffusionOperator end

abstract type AbstractEnrichment end

struct Enrichment_Off <: AbstractEnrichment end

struct Enrichment_On <: AbstractEnrichment
    order::Int
    threshold::Float64
    directional::Bool
end

Base.@kwdef struct DivEtaGrad_WLSQ{DD <: AbstractDiagonalDominance, ET <: AbstractEnrichment} <: AbstractDiffusionOperator
    order::Int = 2
    eta_scaling::Bool = false
    dd::DD = OneDimensionalCorrection_Default()
    enrichment::ET = Enrichment_Off()
end


"""
    Computes strong form div eta grad operator using a point cloud.
    eta_scaling:
    false -> no scaling
    true  -> scaling μ = log(η) with exp(μ) rescaling
"""
function div_eta_grad_row(i, pc::PointCloud{2}, method::DivEtaGrad_WLSQ; kwargs...)
    if method.order < 2
        error("Div eta grad operator should be at least of second order!")
    end

    neighbors = pc.p.neighbors[i]
    cgrad = pc.p.c_gradient[i]
    eta  = pc.p.eta[neighbors]
    mue  = pc.p.mue[neighbors]

    r    = zeros(nfuns(method.order, 2))
    K, W = localmatrices(pc.p.x[i], pc.p.h[i], neighbors, method.order, pc; kwargs...)

    if method.eta_scaling == false
        r[2:3] = cgrad * eta
        r[4] = 2eta[1]
        r[6] = 2eta[1]
    elseif method.eta_scaling == true
        r[2:3] = cgrad * mue
        r[4] = 2.0
        r[6] = 2.0
    end

    rescale_factor = if method.eta_scaling == false
        1.0
    elseif method.eta_scaling == true
        exp(mue[1])
    end

    enrichment = method.enrichment
    K̃, r̃ = enrichment_matrices(i, pc, enrichment)

    while rank([K; K̃]) < size(K, 1) + size(K̃, 1)
        enrichment = Enrichment_On(enrichment.order-1, enrichment.threshold, enrichment.directional)
        K̃, r̃ = enrichment_matrices(i, pc, enrichment)
    end

    if method.eta_scaling
        r̃ /= rescale_factor
    end

    K = [K; K̃]
    r = [r; r̃]

    return rescale_factor * leastsquares(K, W, r, method.dd)

end

function enrichment_matrices(i, pc, ::Enrichment_Off)
    return zeros(0, length(pc.p.neighbors[i])), zeros(0)
end

function enrichment_matrices(i, pc, enrichment::Enrichment_On)
    neighbors = pc.p.neighbors[i]
    clap  = pc.p.c_laplace[i]
    cgrad = pc.p.c_gradient[i]
    eta   = pc.p.eta[neighbors]
    mue   = pc.p.mue[neighbors]

    (; order, threshold, directional) = enrichment

    grad_eta = cgrad * eta
    min_eta, max_eta = extrema(eta)
    if max_eta / min_eta < threshold || enrichment.order < 0
        return enrichment_matrices(i, pc, Enrichment_Off())
    end

    if directional
        n⃗ = SVector{2, Float64}(normalize(cgrad * mue))
        t⃗ = perp(n⃗)

        K = zeros(nfuns(order, 2), length(neighbors))
        r = zeros(nfuns(order, 2))

        for (k, j) = enumerate(neighbors)
            dx = pc.p.x[j] - pc.p.x[i]

            dn = dot(dx, n⃗)
            dt = dot(dx, t⃗)

            K[1, k] = 1.0 / eta[k]

            if order >= 1
                K[2, k] = dn / eta[k]
                K[3, k] = dt / eta[k]
            end

            if order >= 2
                K[4, k] = dn^2 / eta[k]
                K[5, k] = dn * dt / eta[k]
                K[6, k] = dt^2
            end
        end

        r[1] = -dot(clap, mue)
        if order >= 1
            r[2] = -dot(cgrad * mue, n⃗)
        end

        if order >= 2
            r[4] = 2.0
            r[6] = 2.0
        end

    else
        K, _ = localmatrices(pc.p.x[i], pc.p.h[i], neighbors, order, pc)
        r = zeros(nfuns(order, 2))

        for j = axes(K, 2)
            for i = axes(K, 1)
                K[i, j] /= eta[j]
            end
        end

        r[1] = -dot(clap, mue)
        if order >= 1
            r[2:3] = -cgrad * mue
        end
        if order >= 2
            r[4] = 2.0
            r[6] = 2.0
        end
    end
    return K, r
end

# function div_eta_grad_row(i, pc, method::DivEtaGradRow{:flux_conservation})
#     neighbors = pc.p.neighbors[i]

#     cx = pc.operators[:x][i, neighbors]
#     cy = pc.operators[:y][i, neighbors]

#     eta = pc.p.eta[neighbors]

#     dir = normalize(SVector(dot(cx, eta), dot(cy, eta)))

#     ptsL = Int[]
#     ptsR = Int[]

#     ptsLloc = Int[]
#     ptsRloc = Int[]

#     for (k,j) = enumerate(neighbors)
#         dv = pc.p.x[j] - pc.p.x[i]
#         dp = dot(dv, dir)
#         dp <= 0 && (push!(ptsL, j), push!(ptsLloc, k))
#         dp >= 0 && (push!(ptsR, j), push!(ptsRloc, k))
#     end

#     etaLs = eta[ptsLloc]
#     etaRs = eta[ptsRloc]

#     nL = length(ptsL)
#     nR = length(ptsR)

#     etaL = exp(sum(log, etaLs) ./ nL)
#     etaR = exp(sum(log, etaRs) ./ nR)

#     # println("i = $i: etaL = $etaL, etaR = $etaR")

#     cxL = directional_derivative_row(dir, pc.p.x[i], pc.p.h[i], ptsL, method.order, pc, method.dd)
#     cxR = directional_derivative_row(dir, pc.p.x[i], pc.p.h[i], ptsR, method.order, pc, method.dd)

#     v = zeros(precision(pc), length(neighbors))

#     v[ptsLloc] += etaL * cxL
#     v[ptsRloc] -= etaR * cxR

#     return v[1] < zero(v[1]) ? v : -v
# end

# function div_eta_grad_row(i, pc, ::DivEtaGradRow{:weak_voronoi})
#     edgelength, area = voronoi(i, pc)
#     neighbors = pc.p.neighbors[i]
#     eta = pc.p.eta[neighbors]

#     v = zeros(precision(pc), length(neighbors))

#     for (k, j) = enumerate(neighbors)
#         edgelength[k] == 0.0 && continue
#         # etaij = (eta[1] + eta[k]) / 2
#         etaij = 2 * (eta[1] * eta[k]) / (eta[1] + eta[k])
#         # etaij = sqrt(eta[1] * eta[k])
#         W = etaij * ( edgelength[k] / norm(pc.p.x[i] - pc.p.x[j]) )
#         v[1] -= W
#         v[k] += W
#     end

#     return v / area
# end

struct DivEtaGrad_ScaledLaplacian{LAP <: AbstractLaplaceOperator} <: AbstractDiffusionOperator
    etaMean::Symbol
    laplacian::LAP
end

eta_mean(i, pc, ::Val{:minimum}) = map(x -> min(pc.p.eta[i], x), pc.p.eta[pc.p.neighbors[i]])
eta_mean(i, pc, ::Val{:maximum}) = map(x -> max(pc.p.eta[i], x), pc.p.eta[pc.p.neighbors[i]])
eta_mean(i, pc, ::Val{:arithmetic_mean}) = map(x -> (x + pc.p.eta[i]) / 2, pc.p.eta[pc.p.neighbors[i]])
eta_mean(i, pc, ::Val{:harmonic_mean})   = map(x -> 2 / (1 / pc.p.eta[i] + 1 / x), pc.p.eta[pc.p.neighbors[i]])
eta_mean(i, pc, ::Val{:geometric_mean})  = map(x -> sqrt(x * pc.p.eta[i]), pc.p.eta[pc.p.neighbors[i]])
eta_mean(i, pc, ::Val{:center})  = pc.p.eta[i]

function gradient_weno(i, U, pc; ϵ = 1e-5, r = 2)
    ∇U = SVector{dimension(pc), GFDM.precision(pc)}[]
    neighbors = pc.p.neighbors[i]
    xi = pc.p.x[i]
    Ui = U[i]
    A = zero(MMatrix{2, 2, GFDM.precision(pc)})
    b = zero(MVector{2,    GFDM.precision(pc)})
    for j = 2:length(neighbors)
        for k = j+1:length(neighbors)
            xj = pc.p.x[neighbors[j]]
            xk = pc.p.x[neighbors[k]]

            A[1,:] = xj - xi
            A[2,:] = xk - xi

            det(A) == 0 && continue

            b[1] = U[neighbors[j]] - Ui
            b[2] = U[neighbors[k]] - Ui

            ∇Ujk = A \ b

            push!(∇U, ∇Ujk)
        end
    end

    weights = (norm.(∇U) .+ ϵ).^(-r)
    sumweights = sum(weights)

    w = weights / sumweights

    return sum(w .* ∇U)
end

function eta_mean(i, pc, ::Val{:taylor_expansion})
    neighbors = pc.p.neighbors[i]
    eta       = pc.p.eta[neighbors]
    DX        = pc.p.x[neighbors] .- (pc.p.x[i], )
    grad_eta  = pc.p.c_gradient[i] * eta
    eta_ij    = map(k -> eta[1] + 0.5 * dot(grad_eta, DX[k]), eachindex(neighbors))

    return eta_ij
end

function eta_mean(i, pc, ::Val{:skew_taylor})
    neighbors = pc.p.neighbors[i]
    eta       = pc.p.eta[neighbors]
    DX        = pc.p.x[neighbors] .- (pc.p.x[i], )
    grad_eta  = pc.p.c_gradient[i] * eta
    eta_ij    = map(k -> eta[k] - 0.5 * dot(grad_eta, DX[k]), eachindex(neighbors))

    return eta_ij
end

function eta_mean(i, pc, ::Val{:hermite_interpolation})
    neighbors = pc.p.neighbors[i]
    etas      = pc.p.eta[neighbors]
    DX        = pc.p.x[neighbors] .- (pc.p.x[i], )

    # gradients = map(i -> gradient_weno(i, pc.p.eta, pc), neighbors)
    gradients = map(j -> pc.p.c_gradient[j] * pc.p.eta[pc.p.neighbors[j]], neighbors)

    eta_ijs = map(
        k -> 0.5 * (etas[1] + etas[k]) + 0.125 * dot(gradients[1] - gradients[k], DX[k]),
        eachindex(neighbors)
    )

    # for (k, eta_ij) = enumerate(eta_ijs)
    #     mink = min(etas[1], etas[k])
    #     maxk = max(etas[1], etas[k])

    #     if eta_ij > maxk
    #         eta_ijs[k] = maxk
    #         println("Correcting above!")
    #     elseif eta_ij < mink
    #         eta_ijs[k] = mink
    #         println("Correcting below!")
    #     end
    # end

    return eta_ijs
end

function div_eta_grad_row(i, pc, method::DivEtaGrad_ScaledLaplacian; kwargs...)
    etamean = eta_mean(i, pc, Val(method.etaMean))
    clap    = laplace_row(i, pc, method.laplacian; kwargs...)

    cdeg    = etamean .* clap
    cdeg[1] = -sum(cdeg[2:end])

    return cdeg
end

# function div_eta_grad_row(i, pc, ::DivEtaGradRow{:weak_centroid})
#     neighbors   = pc.p.neighbors[i]
#     n_neighbors = length(neighbors)

#     eta = pc.p.eta[neighbors]

#     area = pc.p.dV[i]
#     delaunay = pc.cells[pc.p.simplices[i]]

#     edgelength = zeros(precision(pc), n_neighbors)

#     for tri = delaunay
#         centroid = sum(pc.p.x[[tri]]) / 3
#         edge_out = setdiff(tri, i)

#         for j = edge_out
#             pmid = (pc.p.x[i]+ pc.p.x[j]) / 2

#             k = findfirst(==(j), neighbors)
#             edgelength[k] += norm(pmid - centroid)
#         end
#     end

#     v = zeros(precision(pc), n_neighbors)

#     for k = eachindex(neighbors)
#         edgelength[k] == 0.0 && continue
#         j = neighbors[k]
#         etaij = (eta[1] + eta[k]) / 2
#         W = (etaij * edgelength[k]) / norm(pc.p.x[i] - pc.p.x[j])
#         v[1] -= W
#         v[k] += W
#     end

#     return v / area
# end

# function div_eta_grad_row(i, pc, method::DivEtaGradRow{:gfvm})
#     edgelength, area = voronoi(i, pc)
#     neighbors = pc.p.neighbors[i]
#     eta = pc.p.eta[neighbors]

#     c = zeros(length(neighbors))

#     for (k, j) = enumerate(neighbors)
#         (edgelength[k] ≈ 0.0 || j == k) && continue
#         etaij = 2 * (eta[1] * eta[k]) / (eta[1] + eta[k])

#         nij = normalize(pc.p.x[j] - pc.p.x[i])
#         xij = (pc.p.x[i] + pc.p.x[j]) / 2
#         γik = directional_derivative_row(nij, xij, pc.p.h[i], pc.p.neighbors[i], method.order, pc, method.dd)

#         if γik[1] >= 0 || any(<(0), γik[2:end])
#             error("Found non diagonally dominant γik for point $i and neighbor $j")
#         end

#         c += edgelength[k] * etaij * γik
#     end

#     if area ≈ 0
#         error("Area for point $i almost zero, area = $area")
#     end

#     return c / area
# end
