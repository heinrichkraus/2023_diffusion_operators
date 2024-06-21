"""
Wraps rectangular point cloud in the way that opposite boundaries are connected.
Sets new "data" field such all previous computations are lost!
"""
function wrap!(pc::PointCloud, labelL, labelR)
    println("Wrapping point cloud with $labelL and $labelR labels.")
    indicesL = findall(==(labelL), pc.p.label)
    indicesR = findall(==(labelR), pc.p.label)

    if isempty(indicesL) || isempty(indicesR)
        println("Labels are empty. Point cloud cannot be wrapped.")
        return
    end

    # compute minimal distance between points from either label
    δh = minimum(pc.p.h) / 2.5
    mindistance = min(length(pc) * maximum(pc.p.h), 1e99)
    for i = indicesL, j = indicesR
        dist = norm(pc.p.x[i] - pc.p.x[j])
        if dist < mindistance
            mindistance = dist
        end
    end

    L = mindistance + δh

    neighborlist = neighbors(pc)

    # nL = -nR
    nL = pc.p.n[indicesL[1]] #pc.normal[indicesL[1]]
    nR = pc.p.n[indicesR[1]] #pc.normal[indicesR[1]]

    pointsL = unique(reduce(vcat, neighborlist[indicesL]))
    pointsR = unique(reduce(vcat, neighborlist[indicesR]))

    pdataL = StructArray(zeros(eltype(particles(pc)), length(pointsL)))
    pdataR = StructArray(zeros(eltype(particles(pc)), length(pointsR)))

    for (i, k) = enumerate(pointsL)
        pdataL.x[i]        = pc.p.x[k] - L * nL
        pdataL.n[i]        = nR
        pdataL.h[i]        = pc.p.h[k]
        pdataL.partner[i]  = k
        pdataL.label[i]    = labelR
    end

    for (i, k) = enumerate(pointsR)
        pdataR.x[i]       = pc.p.x[k] - L * nR
        pdataR.n[i]       = nL
        pdataR.h[i]       = pc.p.h[k]
        pdataR.partner[i] = k
        pdataR.label[i]   = labelL
    end

    pc.p.label[indicesL] .= Symbol()
    pc.p.label[indicesR] .= Symbol()

    pc.p.ident[indicesL] .= :none
    pc.p.ident[indicesR] .= :none

    # coordsL is appended to Γ_R and coordsR to Γ_L
    append!(particles(pc), pdataL, pdataR)
end