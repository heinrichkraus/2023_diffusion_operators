PATH_TO_MESHFREE = "/m/scratch/hive/kraus/FPMdebug_double_mpi_shm.x"

function generate_pointcloud(pci::PointCloudInfo{shape}, ::Val{:meshfree}) where shape
    wdir = pwd()
    tdir = mktempdir(wdir)
    cd(tdir)

    # generate boundary points
    bndpoints = bnd_points(pci)

    # generate mesh.FDNEUT
    fdneutname, fdneutpath = writefdneut(bndpoints)

    # generate UCV.dat and CV.dat
    writeucv(fdneutpath, 2 * pci.data.h, fdneutname)
    writecv(length(bndpoints[1]))

    if !(@isdefined PATH_TO_MESHFREE)
        @warn "PATH_TO_MESHFREE has to be set!"
        cd(wdir)
        rm(tdir; recursive = true)
        return nothing
    end

    # run script
    run(`xterm -e $PATH_TO_MESHFREE`)

    fname = "ascii/pointcloud_00001.dat"
    try
        asciifile = open(fname)
        pc = pointcloud_from_ascii(CSV.File(asciifile, normalizenames = true))
        close(asciifile)
        return pc
    finally
        @warn "File $fname could not be read!"
        cd(wdir)
        rm(tdir; recursive = true)
        return nothing
    end

end

bnd_points(pci::PointCloudInfo{:square}) = mani_rectangle(-1.0, 1.0, -1.0, 1.0, pci.data.h)
bnd_points(pci::PointCloudInfo{:circle}) = mani_circle(SVector(0.0, 0.0), 1.0, pci.data.h)

function pointcloud_from_ascii(csvfile)
    pointdata = StructArray(zeros(Particle{2, Float64}, length(csvfile)))
    for (i, row) = enumerate(csvfile)
        @unpack x, y, z, nx, ny, nz, h, kob = row

        ident = if kob == 1
            :none
        elseif kob == 11
            :free_surface
        else
            :wall
        end

        pointdata[i] = Particle{2, Float64}(; x = SVector(x, y), n = SVector(nx, ny), h, ident)
    end
    return PointCloud(pointdata)
end