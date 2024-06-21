# this part of the software sucks

function correct_nodes_voronoi!(pc, correction_weight=(n) -> 1.0; iterations=1, prepared_timestep=false, verbose=false)
    verbose && println("Performing corrections...")
    prepared_timestep || prepare_timestep!(pc)
    for n = 1:iterations
        centroid = zeros(eltype(pc.p.x), length(pc))
        for i = eachindex(pc)
            verbose && println("$i/$(length(pc))")
            if isboundary(pc.p[i])
                centroid[i] = pc.p.x[i]
                continue
            end

            p = pc.p[i]
            (; neighbors, x) = p

            vc = VoronoiCell(i, pc)
            _, vi = voronoi(i, pc)
            for (k, j) = enumerate(neighbors)
                j == i && continue
                iszero(sum(abs, vc.point_mapping[:, k])) && continue
                c1  = vc.edge_point[vc.point_mapping[1, k]]
                c2  = vc.edge_point[vc.point_mapping[2, k]]
                dx  = normalize(pc.p.x[j] - x)
                xij = (c1 + c2) / 2
                sij = norm(c2 - c1)
                centroid[i] += sij * (xij.^2 .* dx)
            end
            centroid[i] /= 2vi

        end
        dx = centroid - pc.p.x
        pc.p.x[:] += correction_weight(n) * dx
        prepare_timestep!(pc)
    end
end

correct_nodes_voronoi!(pc, correction_weight::Number; kwargs...) = correct_nodes_voronoi!(pc, n -> correction_weight; kwargs...)
