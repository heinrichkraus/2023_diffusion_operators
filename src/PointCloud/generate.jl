
struct PointCloudInfo{shape}
    data
end

PointCloudInfo{shape}(; kwargs...) where shape = PointCloudInfo{shape}(values(kwargs))
shapeof(::PointCloudInfo{shape}) where shape = string(shape)

Square(h, type = :uniform) = PointCloudInfo{:square}(h = h, type = type)
Circle(h) = PointCloudInfo{:circle}(h = h, type = :meshfree)

perp(x::SVector{2, Tv}) where Tv <: Number = SVector(-x[2], x[1])

"""
Loads a uniform rectangular point cloud with corner points (xmin, xmax) and (ymin, ymax) and point spacing Δx. \n
The point cloud can be randomized by setting the parameter δ. The point cloud will be randomized with δ * Δx
"""
function rectangle(xmin::T, xmax::T, ymin::T, ymax::T, Δx::T, δ::T = zero(T); ep::Bool=true) where T <: Real
    println("Setting up point cloud with corner points $((xmin,ymin)), $((xmax,ymax)) and Δx = $Δx.")

    nx = convert(Int, round((xmax-xmin)/Δx)) + 1
    ny = convert(Int, round((ymax-ymin)/Δx)) + 1

    x = range(xmin, xmax, length = nx)
    y = range(ymin, ymax, length = ny)

    Tp = Particle{2, T}

    npoints = length(x) * length(y)
    pd = StructArray(zeros(Tp, npoints))
    pd.h .= 2.5Δx

    nL = SVector{2, T}(-1, 0)
    nR = SVector{2, T}( 1, 0)
    nT = SVector{2, T}( 0, 1)
    nB = SVector{2, T}( 0,-1)

    # set up structures
    count = 1
    for i = eachindex(x), j = eachindex(y)
        ((i == 1 && (j == 1 || j == ny) || i == nx && (j == 1 || j == ny)) && !ep) && continue
        pd.x[count] = SVector(x[i], y[j])

        # setpoint!(pd[count], SVector(x[i], y[j]))

        if i == 1 || i == nx || j == 1 || j == ny
            pd.ident[count] = :wall # here should be geometry or something like that...
            # geometry points can be ignored for computations
            if i == 1 && (j == 1 || j == ny)
                pd.label[count] = :left
                pd.n[count]     = nL
            elseif i == nx && (j == 1 || j == ny)
                pd.label[count] = :right
                pd.n[count]     = nR
            else
                pd.ident[count] = :wall
                if i == 1
                    # left point
                    pd.n[count]     = nL
                    pd.label[count] = :left
                elseif i == nx
                    # right point
                    pd.n[count]     = nR
                    pd.label[count] = :right
                elseif j == 1
                    # bottom point
                    pd.n[count]     = nB
                    pd.label[count] = :bottom
                elseif j == ny
                    # top point
                    pd.n[count]     = nT
                    pd.label[count] = :top
                end
            end
        end

        count += 1

    end

    # randomize points within bounds
    if δ != 0.0
        σ = [-δ * Δx, δ * Δx]
        for i = eachindex(pd)
            if pd.label[i] == Symbol()
                δp = rand(σ) * rand(SVector{2, T})
                p̂  = pd.x[i] + δp
                if xmin < p̂[1] < xmax && ymin < p̂[2] < ymax
                    pd.x[i] = p̂
                end
            end
        end
    end

    println("Generated point cloud with $(length(pd)) points!")

    return PointCloud(pd)
end

rectangle(xmin, xmax, ymin, ymax, Δx, δ = 0; kwargs...) = rectangle(promote(xmin, xmax, ymin, ymax, Δx, δ)...; kwargs...)

# function triangle(p1::SVector{2}, p2::SVector{2}, p3::SVector{2}, Δx::Real, δ::Real)
#     d1 = p2 - p1
#     d2 = p3 - p2
#     d3 = p1 - p3

#     m1 = (p1 + p2) / 2
#     m2 = (p2 + p3) / 2
#     m3 = (p3 + p1) / 2

#     nb1 = convert(Int, round(norm(d1) / Δx))
#     nb2 = convert(Int, round(norm(d2) / Δx))
#     nb3 = convert(Int, round(norm(d3) / Δx))

#     h1 = 1 / nb1
#     h2 = 1 / nb2
#     h3 = 1 / nb3

#     points = StructArray(Particle{2, Float64}[])
#     for i = 1:nb1
#         push!(points, Particle(p1 + i * h1 * d1;
#                                 bnd_label = "first",
#                                 bnd_ident = ident_wall))
#     end

#     for i = 1:nb2
#         push!(points, Particle(p2 + i * h2 * d2;
#                                 bnd_label = "second",
#                                 bnd_ident = ident_wall))
#     end

#     for i = 1:nb3
#         push!(points, Particle(p3 + i * h3 * d3;
#                                 bnd_label = "third",
#                                 bnd_ident = ident_wall))
#     end

#     points.h .= 2.5Δx


#     n1 = -normalize(perp(d1))
#     n2 = -normalize(perp(d2))
#     n3 = -normalize(perp(d3))

#     ϕ(x) = max(dot(x - m1, n1), dot(x - m2, n2), dot(x - m3, n3))

#     xmin = min(p1[1], p2[1], p3[1])
#     xmax = max(p1[1], p2[1], p3[1])
#     ymin = min(p1[2], p2[2], p3[2])
#     ymax = max(p1[2], p2[2], p3[2])

#     pc = rectangle(xmin, xmax, ymin, ymax, Δx, δ)
#     pc.bnd_label[:] .= "none"

#     keep = ones(Bool, length(pc))
#     for i = eachindex(pc)
#         if ϕ(pc.p.x[i]) >= -Δx / 3
#             keep[i] = false
#         end
#     end

#     particles(pc) = particles(pc)[keep]
#     append!(particles(pc), points)

#     return pc

# end

function project_points_on_manifold(p_init, f, ∇f; tol = 1e-6, nb_maxiter = 50)
    pc = PointCloud(StructVector(p_init))

    f0 = f.(pc.p)
    err = maximum(abs.(f0))

    # println("Initial error: $err")

    itercount = 0
    while err >= tol && itercount < nb_maxiter
        itercount += 1

        # Newton step
        ∇f0 = ∇f.(pc.p)
        pc.p.n[:] .= ∇f0
        df0 = dot.(∇f0, pc.p.n)
        absf0 = abs.(f0)

        num_nonzero = findall(>=(tol), absf0)
        den_nonzero = findall(!=(0.0), df0)
        nonzero = intersect(den_nonzero, num_nonzero)

        pc.p.x[nonzero] .-= (f0[nonzero] ./ df0[nonzero]) .* pc.p.n[nonzero]

        # Update
        f0  = f.(pc.p)
        err = maximum(abs.(f0))

        # println("Iteration $itercount: error = $err")
    end

    return pc
end

function rectangle_with_alignment(
    xmin, xmax, ymin, ymax,
    Δx,
    pts_init,
    f, ∇f,
    δ = 0.0;
    ϵ = Δx / 3, gap = true
)

    pc1 = rectangle(xmin, xmax, ymin, ymax, Δx, δ)
    pc2 = project_points_on_manifold(pts_init, f, ∇f)


    # filter points
    keep_point = ones(Bool, length(pc2))
    for i = eachindex(pc2), j = i+1:length(pc2)
        if norm(pc2.p.x[i] - pc2.p.x[j]) < 2ϵ/3
            keep_point[i] = false
        end
    end
    pc2.p = pc2.p[keep_point]

    keep_point = ones(Bool, length(pc1))
    for i1 = eachindex(pc1), i2 = eachindex(pc2)
        if norm(pc1.p.x[i1] - pc2.p.x[i2]) < Δx
            keep_point[i1] = false
        end
    end
    pc1.p = pc1.p[keep_point]

    # generate alignment points
    points = if gap
        ∇fp = normalize(∇f.(pc2.p))
        union(
            [Particle(pc2.p.x[i] - 2Δx * ∇fp[i], h = 2.5Δx) for i = eachindex(pc2)],
            [Particle(pc2.p.x[i] + 2Δx * ∇fp[i], h = 2.5Δx) for i = eachindex(pc2)]
        )
    else
        [Particle(pc2.p.x[i], h = 2.5Δx) for i = eachindex(pc2)]
    end

    append!(pc1.p, points)
    return pc1
end

function rectangle_with_circle(xmin, xmax, ymin, ymax, Δx, r, δ = 0.0; ϵ = Δx / 3)
    # generate points with radius r - ϵ and r + ϵ
    P(r, φ) = r * SVector(cos(φ), sin(φ))
    Φ = range(0, 2π - 0.0001, step = acos((2r^2 - Δx^2) / 2r^2))

    pts_in  = [Particle(P(r - ϵ, φ), h = 2.5Δx) for φ ∈ Φ]
    pts_out = [Particle(P(r + ϵ, φ), h = 2.5Δx) for φ ∈ Φ]

    pc = rectangle(xmin, xmax, ymin, ymax, Δx, δ)

    keep = [true for _ = eachindex(pc)]
    for i = eachindex(pc)
        if r - 2ϵ <= norm(pc.p.x[i]) <= r + 2ϵ
            keep[i] = false
        end
    end
    pc.p = pc.p[keep]
    append!(particles(pc), pts_in)
    append!(particles(pc), pts_out)
    return pc
end

function generate_pointcloud(pci::PointCloudInfo, sname = "")
    pc = generate_pointcloud(pci, Val(pci.data.type))
    !isempty(sname) && save_csv(sname, pc)
    return pc
end

function labelpoints!(pc::PointCloud{2}, ::Val{:square})
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf

    for i = eachindex(pc)
        xmin = pc.p.x[i][1] < xmin ? pc.p.x[i][1] : xmin
        xmax = pc.p.x[i][1] > xmax ? pc.p.x[i][1] : xmax
        ymin = pc.p.x[i][2] < ymin ? pc.p.x[i][2] : ymin
        ymax = pc.p.x[i][2] > ymax ? pc.p.x[i][2] : ymax
    end

    for i = eachindex(pc)
        if pc.p.x[i][1] == xmin
            pc.p.label[i], pc.p.ident[i] = :left, :wall
        elseif pc.p.x[i][1] == xmax
            pc.p.label[i], pc.p.ident[i] = :right, :wall
        elseif pc.p.x[i][2] == ymin
            pc.p.label[i], pc.p.ident[i] = :bottom, :wall
        elseif pc.p.x[i][2] == ymax
            pc.p.label[i], pc.p.ident[i] = :top, :wall
        end
    end
end

generate_pointcloud(pci::PointCloudInfo{:square}, ::Val{:uniform}) = rectangle(-1, 1, -1, 1, pci.data.h)
