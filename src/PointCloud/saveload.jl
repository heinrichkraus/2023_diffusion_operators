function vtk_file(filename::AbstractString, pc::PointCloud{2}, table)
    filename = abspath(filename)

    #TODO: add normal vector outputs
    cells   = broadcast(tri -> MeshCell(VTKCellTypes.VTK_TRIANGLE, tri), pc.cells)
    points  = tocolmatrix(pc.p.x)
    vtkfile = vtk_grid(filename, points, cells)

    # save to vtk file
    for prop = propertynames(table)
        data = getproperty(table, prop)
        if eltype(data) <: Number
            vtkfile[string(prop), VTKPointData()] = Float64.(data)
        end
    end

    return vtkfile
end

# quicksave_vtk(filename, pc::PointCloud) = vtk_save(vtk_file(filename, pc))
function quicksave_vtk(filename, pc::PointCloud)
    return vtk_save(vtk_file(filename, pc, pc.p))
end

function prepare_save_items!(table, t, save_items)
    fs = values(save_items)
    for (k, prop) = enumerate(keys(save_items))
        f = fs[k]
        z = f(table, t)
        setindex!(getproperty(table, prop), z, :)
    end

    return table
end

function TypedTables.Table(pc::PointCloud, save_items::NamedTuple)
    return Table(
        particles(pc),
        # rename(name -> Symbol("debug", "_", name), pc.debug);
        NamedTuple{keys(save_items)}([zeros(length(pc)) for key = keys(save_items)])
        )
    end

function save_csv(filename::String, pc::PointCloud{2})
    mkpath(dirname(filename))
    pts = tocolmatrix(pc.p.x)
    nml = tocolmatrix(pc.p.n)
    CSV.write(filename,
        (
            x = view(pts, 1, :), y = view(pts, 2, :),
            nx = view(nml, 1, :), ny = view(nml, 2, :),
            h = pc.p.h,
            bnd_ident = String.(particles(pc).ident)
        )
    )
end

function load_csv(filename::String)
    csvfile = CSV.File(filename, types = Dict(:x => Float64, :y => Float64, :nx => Float64, :ny => Float64, :h => Float64, :bnd_ident => String))
    npoints = length(csvfile)
    points = StructArray(zeros(Particle{2,Float64}, npoints))

    points.x .= SVector.(csvfile.x, csvfile.y)
    points.n .= SVector.(csvfile.nx, csvfile.ny)
    points.h .= csvfile.h
    points.ident .= Symbol.(csvfile.bnd_ident)

    for i = eachindex(points)
        if points.ident[i] == Symbol("0")
            points.ident[i] = :none
        elseif points.ident[i] == Symbol("1")
            points.ident[i] = :wall
        else
            error("Unknown IDENT!")
        end
    end

    return PointCloud(points)
end