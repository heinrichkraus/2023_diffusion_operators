function neighbors!(pc::PointCloud; filterdomain::Bool = true)
    tree = KDTree(pc.p.x)

    for i in eachindex(pc)
        tmp  = inrange(tree, pc.p.x[i], pc.p.h[i])
        keep = ones(Bool, length(tmp))

        if filterdomain
            for (l, j) = enumerate(tmp)
                if pc.p.domain[i] != pc.p.domain[j]
                    keep[l] = false
                end
            end
        end

        # sort points by distance to xi => xi is stored at first position
        dr = norm.(pc.p.x[tmp[keep]] .- (pc.p.x[i], ))
        sp = sortperm(dr)

        pc.p.neighbors[i] = tmp[keep][sp]
    end
end

function neighbors(pc::PointCloud)
    neighbors!(pc)
    return pc.p.neighbors
end