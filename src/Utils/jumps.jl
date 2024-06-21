@enum Jumps begin
    jumps_smooth
    jumps_smooth_local
    jumps_gradient
end

function findjumps(vo, vs, pc::PointCloud, σ::Real, ::Val{jumps_smooth})
    return abs.(vo - vs) .> σ * max(maximum(vo) - minimum(vo), 1e-8)
end

function testjumpident_dd(M, pc, tol)
    is_not_dd = zeros(Int, length(pc))
    value = zeros(length(pc))
    for i = eachindex(pc)
        c = M[i, pc.p.neighbors[i]]
        c[1] != M[i,i] && error("")
        rowsum = sum(abs.(c[2:end])) - abs(c[1])
        if  rowsum > tol
            is_not_dd[i] = 1
            value[i] = rowsum
        end
    end
    return is_not_dd, value
end

function testjumpident_coeffs(M, pc)
    # test where diagonal dominance is not satisfied
    is_dd = zeros(Int, length(pc))
    value = zeros(length(pc))
    for i = eachindex(pc)
        # isboundary(pc.p[i]) && continue
        c = M[i, pc.p.neighbors[i]]
        if c[1] >= 0
            value[i] = c[1]
            is_dd[i] = 1
        elseif any(<(0), c[2:end])
            value[i] = abs(minimum(c[2:end]))
            is_dd[i] = 2
        end
    end
    return is_dd, value
end

function testjumpident_gradient(v, pc)
    pc.debug.function_gradient_norm = sqrt.((pc.operators[:x] * v).^2 + (pc.operators[:y] * v).^2)
    pc.debug.hasjump_gradient = zeros(Int, length(pc))
    for i = eachindex(pc)
        neighbors = pc.p.neighbors[i]
        df = abs.(v[neighbors] .- v[i])
        dgf = pc.debug.function_gradient_norm[neighbors]
        if maximum(df) > pc.p.h[i] * maximum(dgf)
            pc.debug.hasjump_gradient[i] = 1
        end
    end
end

function testjumpident_ratio(vo, pc, σ)
    hasjump_ratio = zeros(Int, length(pc))
    ratio         = zeros(length(pc))

    for i = eachindex(pc)
        neighbors = pc.p.neighbors[i]

        maxval = maximum(vo[neighbors])
        minval = minimum(vo[neighbors])

        ratio[i] = maxval / minval

        if ratio[i] > σ
            hasjump_ratio[i] = true
        end
    end
    return ratio, hasjump_ratio
end

function findjumps(vo, vs, pc::PointCloud, σ::Real, ::Val{jumps_smooth_local})
    hasjump = zeros(Bool, length(pc))
    DV      = zeros(precision(pc), length(pc))
    for i = eachindex(pc)
        neighbors = pc.p.neighbors[i]
        voloc = vo[neighbors]
        # vsloc = vs[neighbors]
        # dvo   = maximum(voloc) - minimum(voloc)
        # dvs   = maximum(vsloc) - minimum(vsloc)
        # dvso  = maximum(vsloc) - minimum(voloc)
        # dvos  = maximum(voloc) - minimum(vsloc)
        # dv    = σ * max(dvo, dvs, dvso, dvos)
        dv    = σ * max(maximum(voloc) - minimum(voloc), 1e-8)
        DV[i] = dv
        if abs(vo[i] - vs[i]) > σ * dv
            hasjump[i] = true
        end
    end

    return hasjump
end

function findjumps(vo, vs, pc::PointCloud, σ::Real, ::Val{jumps_gradient})
    grad = SVector.(pc.operators[:x] * vs, pc.operators[:y] * vs)
    return norm.(grad) .> σ * max(maximum(vo) - minimum(vo), 1e-8)
end