""" Requires points to be sorted counter-clockwise """
function writefdneut(points::AbstractMatrix)
    dim  = size(points, 1)
    npts = size(points, 2)
    pts  = points

    @assert dim == 2 # full 3D point clouds should be generated with different tools...

    # ncon = size(connectivity, 2)
    # con  = connectivity

    name = Random.randstring('a':'z', 10)
    filename =  name * ".FDNEUT"
    file = open(filename, "w")

    println(file, "** FIDAP NEUTRAL FILE")
    println(file, filename)
    println(file, "Version ???")
    println(file, Dates.format(Dates.now(), "dd.mm.yyyy, HH:MM:SS"))

    println(file, "    NO. OF NODES    NO. ELEMENTS  NO. ELT GROUPS           NDFCD           NDFVL")
    @printf(file, "%15u %15u %15u %15u %15u\n", npts, npts, 1, 2, 2)

    println(file, "NODAL COORDINATES")
    for i = 1:npts
        @printf(file, "%8u ", i)
        for j = 1:dim
            @printf(file, " %16.16e", pts[j,i])
        end
        @printf(file, "\n")
    end

    println(file, "ELEMENT GROUPS")
    @printf(file, "%-10s%5i", "GROUP:", 1)
    @printf(file, "%-10s%10i", " ELEMENTS:", npts)
    @printf(file, "%-10s%10i", " NODES:", dim)
    @printf(file, "%-10s%5i", " GEOMETRY:", 0)
    @printf(file, "%-6s%4i", " TYPE:", 15)
    println(file)
    println(file, "ENTITY NAME: $name")
    for i = 1:npts
        @printf(file, "%8u", i)
        if i < npts
            @printf(file, " %8u", i+1)
            @printf(file, " %8u", i)
        else
            @printf(file, " %8u", 1)
            @printf(file, " %8u", i)
        end
        # for j = 1:dim
            # @printf(file, " %8u", con[j,i])
        # end
        @printf(file, "\n")
    end

    close(file)
    return name, filename
end
writefdneut(points::AbstractVector) = writefdneut(reduce(hcat, points))



# rad = 1.0
# h   = 0.1

# γ(t) = SVector(0.0, 0.0) + rad * SVector(cospi(t), sinpi(t))

# tstart = 0.0
# tend   = 2.0
# step   = 2 / pi * asin(h / 2rad)
# len    = Int(round( (tend - tstart) / step ))

# times = range(tstart, tend, length = len)

# npts  = length(times) - 1

# points       = zeros(SVector{2,Float64}, npts)
# connectivity = zeros(SVector{2,Int}, npts)
# for i = 1:npts
#     points[i] = γ(times[i])
#     connectivity[i] = i < npts ? SVector(i, i+1) : SVector(i, 1)
# end

# writefdneut("circle", "Circle$h", points, connectivity)