isseparated(pc::PointCloud) = maximum(pc.p.domain) != minimum(pc.p.domain)

function domain_decomposition!(pc::PointCloud, color, crit)
    length(pc) != length(color) && error("Length of point cloud and color have to be the same!")

    isseparated(pc) && @warn "Point cloud has already been separated."

    # find domain IDs
    domains  = ones(Int, length(pc))
    ndomains = length(crit) + 1

    for k = eachindex(crit)
        for i = eachindex(color)
            if color[i] >= crit[k]
                domains[i] += 1
            end
        end
    end

    # check if domain IDs are unique
    for k = 1:ndomains + 1
        if count(==(k), domains) == 0
            # domain is empty -> reduce all higher indices by one
            for i = eachindex(pc)
                if domains[i] > k
                    domains[i] -= 1
                end
            end
        end
    end

    pc.p.domain .= domains
end

