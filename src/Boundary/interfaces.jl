function interfaces!(pc::PointCloud{2, Tv}) where {Tv}
    Tp = SVector{2, Float64}

    neighbors!(pc, filterdomain = false)
    isempty(pc.cells) && delaunay!(pc)

    # assuming that each domain ID between min and max has some points
    ndomains = length(unique(pc.p.domain))

    if ndomains == 1
        @warn "Point cloud has to be separated before interfaces can be computed."
        return
    end

    # find adjacency of domains and points that are close to an interface
    domainadjacency = zeros(Bool, ndomains, ndomains)
    nearinterface   = spzeros(Bool, length(pc), ndomains)

    for i = eachindex(pc)
        pc.p.ident[i] != :none && continue
        for j = pc.p.neighbors[i]
            if pc.p.domain[i] != pc.p.domain[j]
                domainadjacency[pc.p.domain[i], pc.p.domain[j]] = true
                nearinterface[i, pc.p.domain[j]] = true
            end
        end
    end

    atinterface = zeros(Bool, length(pc))
    directneighbors = spzeros(Vector{Int}, length(pc))

    for i = eachindex(pc)
        if any(isequal(true), nearinterface[i,:])
            atinterface[i] = true
        end
    end

    # reset all partners
    pc.p.partner .= 0

    for i = eachindex(pc)
        if any(isequal(true), nearinterface[i,:])
            atinterface[i] = true
        end

        if atinterface[i]
            # identify if point is real interface point
            # compute local triangulation
            triangles = pc.cells[pc.p.simplices[i]] # stored in N separate dim + 1 - arrays
            neighbors = unique(reduce(hcat, triangles)) # direct neighbors from triangulation

            directneighbors[i] = neighbors

            nearinterface[i,:] .= false
            for j = neighbors
                if pc.p.domain[i] != pc.p.domain[j]
                    nearinterface[i, pc.p.domain[j]] = true
                end
            end

            n_adjacentdomains = sum(nearinterface[i,:])

            if n_adjacentdomains > 1
                error("Point $i has direct neighbors from $n_adjacentdomains different domains. Use finer point cloud!")
            elseif n_adjacentdomains < 1
                # i is not a boundary point
                nearinterface[i,:] .= false
                atinterface[i]      = false
                continue
            end

            # calculate triangle types
            # triangle_type[k] = n <-> triangle has n points from same phase as i
            triangletype = zeros(Int, length(triangles))
            for k = eachindex(triangles), l = 1:3
                j = triangles[k][l]
                if pc.p.domain[i] == pc.p.domain[j]
                    triangletype[k] += 1
                end
            end

            if count(==(2), triangletype) != 2
                # strange triangulation...
                println("Normal vectors cannot be computed for point $i. Triangle types are $triangletype. Skipping point.")
                nearinterface[i,:] .= false
                atinterface[i]      = false
                continue
            end

            # # === === normal vectors === === #
            # # calculate points that help for normal vector calculation
            j1 = 0
            j2 = 0
            tri1 = 0
            tri2 = 0
            for k = eachindex(triangles)
                triangletype[k] != 2 && continue
                for l = 1:3
                    j = triangles[k][l]
                    if pc.p.domain[i] == pc.p.domain[j] && i != j
                        j1 == 0 ? j1 = j : j2 = j
                        tri1 == 0 ? tri1 = k : tri2 = k
                    end
                end
            end

            nvecs = [normalize(perp(pc.p.x[i] - pc.p.x[j1])),
                     normalize(perp(pc.p.x[i] - pc.p.x[j2]))]

            # compute barycenter using points from same domain to orient normal vector
            ptmp = zero(Tp)
            npts = 0

            for k = eachindex(triangles)
                if triangletype[k] >= 2
                    ptmp += sum(pc.p.x[triangles[k]])
                    npts += 3
                end
            end

            ptmp /= npts

            for k = 1:2
                if dot(nvecs[k], ptmp - pc.p.x[i]) > zero(Tv)
                    nvecs[k] *= -one(Tv)
                end
            end
            ptmp = sum(nvecs)

            ptmp = zero(Tp)
            npts = 0
            for k = eachindex(triangles)
                if triangletype[k] <= 2
                    ptmp += normalize(sum(pc.p.x[triangles[k]]) / 3 - pc.p.x[i])
                end
            end
            pc.p.n[i] = normalize(ptmp)
        end
    end

    maxprefpartners = 3
    prefpartners = spzeros(Int, length(pc), maxprefpartners)
    weightstemp  = spzeros(Tv, length(pc), maxprefpartners)

    # === === === preferred partner calculation === === === #
    for i = eachindex(pc)
        if atinterface[i]
            possiblepartners = zeros(Int, length(directneighbors))
            n_possiblepartners = 0

            for j = directneighbors[i]
                if pc.p.domain[i] != pc.p.domain[j]
                    n_possiblepartners += 1
                    possiblepartners[n_possiblepartners] = j
                end
            end

            dist_possiblepartners = zeros(Tv, n_possiblepartners)
            for k = 1:n_possiblepartners
                j = possiblepartners[k]
                xp = perp(pc.p.x[i] - pc.p.x[j])
                dist_possiblepartners[k] = norm(pc.p.x[i] - pc.p.x[j]) *
                (
                    abs(dot(pc.p.n[i], xp)) + abs(dot(pc.p.n[j], xp))
                )
            end

            I = sortperm(dist_possiblepartners)
            k = 0
            weightstemp[i,:] .= Tv(Inf)

            ub = min(maxprefpartners, n_possiblepartners)
            while k < ub
                k += 1
                weightstemp[i,k] = dist_possiblepartners[I[k]]
                prefpartners[i,k] = possiblepartners[I[k]]
            end
        end
    end

    # === === === calculating interface === === === #
    available = zeros(Bool, length(pc))
    for i = eachindex(pc)
        atinterface[i] ? available[i] = true : available[i] = false
    end

    for idom = 1:ndomains, jdom = idom+1:ndomains
        if domainadjacency[idom, jdom]
            n1 = count(==(idom), atinterface.* pc.p.domain)
            n2 = count(==(jdom), atinterface.* pc.p.domain)

            points_idom = zeros(Int, n1)
            points_jdom = zeros(Int, n2)

            n1 = 0
            n2 = 0

            for i = eachindex(pc)
                atinterface[i] || continue
                if pc.p.domain[i] == idom && nearinterface[i, jdom]
                    n1 += 1
                    points_idom[n1] = i
                elseif pc.p.domain[i] == jdom && nearinterface[i, idom]
                    n2 += 1
                    points_jdom[n2] = i
                end
            end

            if n1 < n2
                n = n1
                points = points_idom
                # revert = false
            else
                n = n2
                points = points_jdom
                # revert = true
            end

            # === === === matching points === === === #
            weights = zeros(Tv, n)
            for k = 1:n
                weights[k] = weightstemp[points[k], 1]
            end

            for _ = 1:maxprefpartners
                I = sortperm(weights)
                for l = 1:n
                    i = points[I[l]]

                    available[i] || continue

                    j = prefpartners[i,1]

                    if j == 0 || weights[I[l]] == Tv(Inf)
                        # point has no partner
                        continue
                    elseif !available[j]
                        # partner already taken -> move forward to next best partner
                        for k = 1:maxprefpartners-1
                            prefpartners[i,k] = prefpartners[i,k+1]
                            weightstemp[i,k] = weightstemp[i,k+1]
                        end

                        prefpartners[i,end] = 0
                        weightstemp[i,end] = Tv(Inf)
                        continue
                    end

                    available[i] = false
                    available[j] = false

                    pc.p.partner[i] = j
                    pc.p.partner[j] = i

                    # if revert
                    #     σ = -one(Tv)
                    # else
                    #     σ = one(Tv)
                    # end

                    pc.p.ident[i] = :interface
                    pc.p.ident[j] = :interface

                    prefpartners[i,:] .= 0
                    weightstemp[i,:]  .= zero(Tv)

                    prefpartners[j,:] .= 0
                    weightstemp[j,:]  .= zero(Tv)
                end

                for l = 1:n
                    weights[l] = weightstemp[points[l], 1]
                end
            end
        end
    end
end