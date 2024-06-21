function delaunay!(pc::PointCloud{2, Tv, Data}; checks = false, αtol = deg2rad(10)) where {Tv, Data}
    tio = Triangulate.TriangulateIO()

    tio.pointlist = tocolmatrix(pc.p.x)
    (delaunay, voronoi) = Triangulate.triangulate("Qc", tio)

    ntriangles = size(delaunay.trianglelist, 2)

    triangle_counter = 0

    keeptriangle = ones(Bool, ntriangles)
    triangles = [Int[] for _ = eachindex(pc)]

    for i = 1:ntriangles
        i1, i2, i3 = delaunay.trianglelist[:,i]

        # filter triangles that are almost degenerate
        a = norm(delaunay.pointlist[:,i2] - delaunay.pointlist[:,i1])
        b = norm(delaunay.pointlist[:,i3] - delaunay.pointlist[:,i2])
        c = norm(delaunay.pointlist[:,i1] - delaunay.pointlist[:,i3])

        ρα = (b^2 + c^2 - a^2) / (2 * b * c)
        ρβ = (a^2 + c^2 - b^2) / (2 * a * c)
        ργ = (a^2 + b^2 - c^2) / (2 * a * b)

        if any(>=(1) ∘ abs, (ρα, ρβ, ργ))
            keeptriangle[i] = false
            continue
        end

        α = acos(ρα)
        β = acos(ρβ)
        γ = acos(ργ)

        if min(α, β, γ) < αtol
            keeptriangle[i] = false
            continue
        end

        # compute mapping from point to triangles
        if keeptriangle[i]
            triangle_counter += 1
            push!(triangles[i1], triangle_counter)
            push!(triangles[i2], triangle_counter)
            push!(triangles[i3], triangle_counter)
        end
    end

    if checks
        for i = eachindex(pc)
            tri = triangles[i]
            isempty(tri) && error("Found isolated point at i = $(i)! Cannot resolve this problem yet.")
            unique(delaunay.trianglelist[:,tri]) ⊈ pc.p.neighbors[i] &&
                error("Triangulation of point $i contains points outside of its neighborhood.")
        end
    end

    pc.p.simplices .= triangles
    pc.cells = tovecvec(delaunay.trianglelist[:,keeptriangle])
end

struct VoronoiCell{dim, Tv}
    edge_point::Vector{SVector{dim, Tv}}
    point_mapping::Matrix{UInt32}
end

function VoronoiCell(i, pc::PointCloud{2})
    p = pc.p[i]
    (; simplices, neighbors) = p
    triangles     = pc.cells[simplices]
    circumcenters = zeros(SVector{2, GFDM.precision(pc)}, length(triangles))

    point_to_circ = zeros(UInt32, 2, length(neighbors))
    np_to_circ    = zeros(UInt32, length(neighbors))

    for (k, tri) = enumerate(triangles)
        # compute circumcenters for triangles
        i1, i2, i3 = tri
        circumcenters[k] = circumcenter(pc.p.x[i1], pc.p.x[i2], pc.p.x[i3])

        for (l, j) = enumerate(neighbors)
            j == i && continue
            if j ∈ tri
                np_to_circ[l] += 1
                point_to_circ[np_to_circ[l], l] = k
            end
        end
    end

    # should probably be sorted in some way maybe?
    return VoronoiCell(circumcenters, point_to_circ)
end

function edge_to_tri_mapping(i, pc::PointCloud{2})
    triangles = pc.cells[pc.p.simplices[i]]

    neighbors   = pc.p.neighbors[i]
    n_neighbors = length(neighbors)

    edge2tri  = zeros(Int, 2, n_neighbors)
    nedge2tri = zeros(Int, n_neighbors)

    for (k, tri) = enumerate(triangles)
        # compute connectivity
        for l = tri
            l == i && continue
            m = findfirst(==(l), neighbors) # this can fail when the point cloud does not satisfy Nᵢ ⊂ Sᵢ
            if isnothing(m)
                error("Probably neighborhoods are not set up...")
            end
            nedge2tri[m] += 1
            edge2tri[nedge2tri[m], m] = k
        end
    end

    return edge2tri, nedge2tri
end

function voronoi(i, pc::PointCloud{2})
    triangles = pc.cells[pc.p.simplices[i]]
    circumcenters = zeros(SVector{2, GFDM.precision(pc)}, length(triangles))

    neighbors   = pc.p.neighbors[i]
    n_neighbors = length(neighbors)

    edge2tri, nedge2tri = edge_to_tri_mapping(i, pc)

    for (k, tri) = enumerate(triangles)
        # compute circumcenters for triangles
        i1, i2, i3 = tri
        circumcenters[k] = circumcenter(pc.p.x[i1], pc.p.x[i2], pc.p.x[i3])
    end

    edgelength = zeros(GFDM.precision(pc), n_neighbors)
    area = zero(GFDM.precision(pc))

    for (k, j) = enumerate(neighbors)
        nedge2tri[k] == 0 && continue

        if nedge2tri[k] == 1
            i1 = edge2tri[1,k]
            p1 = circumcenters[i1]
            p2 = (pc.p.x[i] + pc.p.x[j]) / 2
        elseif nedge2tri[k] == 2
            i1 = edge2tri[1,k]
            i2 = edge2tri[2,k]
            p1 = circumcenters[i1]
            p2 = circumcenters[i2]
        else
            error("")
        end

        edgelength[k] = norm(p1 - p2)
        area += edgelength[k] * norm(pc.p.x[i] - pc.p.x[j]) / 4
    end

    return edgelength, area
end

function calculate_normals!(pc::PointCloud{2})
    for i = eachindex(pc)
        !isboundary(pc.p[i]) && continue
        (; x, neighbors) = pc.p[i]
        xb = sum(pc.p.x[neighbors]) / length(neighbors) # control point for normal calculation
        nvec = zero(SVector{2, GFDM.precision(pc)})
        _, nedge2tri = edge_to_tri_mapping(i, pc)
        pc.p.dS[i] = 0.0
        for (k, j) = enumerate(neighbors)
            !isboundary(pc.p[j]) && continue
            nedge2tri[k] != 1 && continue

            ntemp = perp(pc.p.x[j] - pc.p.x[i])
            norm_ntemp = norm(ntemp)
            ntemp = ntemp / norm_ntemp
            pc.p.dS[i] += norm_ntemp / 2
            ntemp *= sign(dot(ntemp, x - xb))
            nvec += ntemp/2
        end
        pc.p.n[i] = normalize(nvec)
    end
end

function simplex_neighbors(i, pc)
    simplices = pc.p.simplices[i]
    return unique(vcat(pc.cells[simplices]...))
end

function surface_volume_voronoi(i, pc::PointCloud{2})
    isboundary(pc.p[i]) || @error "Point $i is not a boundary point!"
    _, area = voronoi(i, pc)
    si = zero(GFDM.precision(pc))

    directneighbors = simplex_neighbors(i, pc)
    directneighbors = intersect(pc.p.neighbors[i], directneighbors) # hopefully for sorting faster
    counter = 0
    for j = directneighbors
        i == j && continue
        (isboundary(pc.p[j]) || isgeometry(pc.p[j])) || continue
        counter = counter + 1
        counter >= 3 && break
        fac = isgeometry(pc.p[j]) ? 1.0 : 0.5
        si += fac * norm(pc.p.x[i] - pc.p.x[j])
    end

    return si, area
end

function circumcenter(p1, p2, p3)
    d12 = p1 - p2
    d23 = p2 - p3
    d31 = p3 - p1

    n12 = norm(d12)
    n23 = norm(d23)
    n31 = norm(d31)

    den = 2 * norm(cross(d12, d23))^2

    α = -n23^2 * dot(d12, d31) / den
    β = -n31^2 * dot(d12, d23) / den
    γ = -n12^2 * dot(d31, d23) / den

    return α * p1 + β * p2 + γ * p3
end

function trianglearea(a::Real, b::Real, c::Real)
    a, b, c = promote(a, b, c)
    s = (a + b + c) / 2
    return sqrt(s * (s-a) * (s-b) * (s-c))
end

function volumes!(pc::PointCloud{2, Tv, Data}; volume_type = :voronoi, kwargs...) where {Tv, Data}
    isempty(pc.cells) && delaunay!(pc)

    pc.p.dS .= zeros(length(pc))
    pc.p.dV  .= zeros(length(pc))

    if volume_type == :delaunay_dual
        for i = eachindex(pc.cells)
            i1, i2, i3 = pc.cells[i]

            a = norm(pc.p.x[i1] - pc.p.x[i2])
            b = norm(pc.p.x[i2] - pc.p.x[i3])
            c = norm(pc.p.x[i3] - pc.p.x[i1])

            area = trianglearea(a, b, c) / 3

            pc.p.dV[i1] += area
            pc.p.dV[i2] += area
            pc.p.dV[i3] += area
        end
    elseif volume_type == :voronoi
        for i = eachindex(pc)
            if isboundary(pc.p[i])
                pc.p.dS[i], pc.p.dV[i] = surface_volume_voronoi(i, pc)
            else
                _, pc.p.dV[i] = voronoi(i, pc)
            end
        end
    else
        error("Unknown volume computation method $volume_type.")
    end

end


function localdelaunay!(pc::PointCloud{2, Tv, Data}) where {Tv, Data}
    @warn "This method is not good..."

    tio = Triangulate.TriangulateIO()
    delaunay = Triangulate.TriangulateIO()

    cells = SVector{3, Float64}[]
    simplices = [Int[] for _ in eachindex(pc)]

    triangle_counter = 0

    for i = eachindex(pc)
        neighbors = pc.p.neighbors[i]
        tio.pointlist = tocolmatrix(pc.p.x[neighbors])

        (delaunay, _) = Triangulate.triangulate("Qc", tio)

        for tri = eachcol(delaunay.trianglelist)
            if any(==(1), tri)
                # this is a triangle containing the central point
                # convert local coordinates to global coordinates
                push!(cells, SVector(neighbors[tri]...))
                triangle_counter = triangle_counter + 1
                push!(simplices[i], triangle_counter)
            end
        end
    end

    pc.cells = cells
    pc.p.simplices .= simplices
end