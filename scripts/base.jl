using Core: Const
using DrWatson
using Revise
@quickactivate :GFDM

using CSV

# saving of convergence results as .csv files
function save_data(
    scriptdir, df::DataFrame;
    format=".csv",
    prefix="",
    suffix=""
)
    dir, filename = splitdir(scriptdir)
    savedir  = joinpath(dirname(dir), "data")
    savename = first(splitext(filename))

    pfx = isempty(prefix) ? "" : prefix*"_"
    sfx = isempty(suffix) ? "" : "_"*suffix

    mkpath(savedir)
    CSV.write(joinpath(savedir, pfx*savename*sfx*format), df)
end

# error calculation
li_error(Uh, U) = norm(Uh - U, Inf) / (maximum(U)-minimum(U))

# saving routine for analytic solutions and diffusivities
function save_testcase(scriptdir, u, λ, h)
    dir, filename = splitdir(scriptdir)
    name = first(splitext(filename))
    hstr = last(split(string(h), "."))
    file = joinpath(dirname(dir), "data", "testcases_$hstr.csv")

    pc = load_pc(h)
    df = if isfile(file)
        df = CSV.read(file, DataFrame)
    else
        df = DataFrame(x=xs(pc), y=ys(pc))
    end

    setproperty!(df, Symbol(name*"_solution"),    u.(pc.p))
    setproperty!(df, Symbol(name*"_diffusivity"), λ.(pc.p))

    mkpath(dirname(file))
    CSV.write(file, df)
end

# scale square point cloud from [-1, 1]² to [0, 1]²
function rescale!(pc)
    pc.p.x[:] *= 0.5
    pc.p.x[:] .+= (SVector(0.5, 0.5), )
    pc.p.h[:] *= 0.5
end

# wrappers for diffusion operators
diffusion_method(method) = DiffusionSystem(Smoothing(), DiffusionOperator_Single(method))
ddo(mean, p) = diffusion_method(DivEtaGrad_ScaledLaplacian(mean, Laplace_WLSQ(; order=p)))
mls(p)       = diffusion_method(DivEtaGrad_WLSQ(p, false, OneDimensionalCorrection_Default(), Enrichment_Off()))

methods_ddo(ps, averages) = (method_row_ddo(p, avg[1], avg[2]) for p=ps, avg=pairs(averages))
methods_mls(ps) = (method_row_mls(p, q) for p=ps, q=(2, 4) if q <= p)

function method_row_ddo(p, avg_name, avg_method)
    name   = Symbol("ddo_", p, "_", avg_name)
    method = ddo(avg_method, p)
    return (; name, method)
end

function method_row_mls(p, q)
    name           = Symbol("mls_", p, "_", q)
    method         = mls(p)
    gradient_order = q
    return (; name, method, gradient_order)
end

# default diffusivity averaging methods
const AVG_DEFAULT = (
    am = :arithmetic_mean,
    hm = :harmonic_mean,
    gm = :geometric_mean,
    hi = :hermite_interpolation,
    te = :taylor_expansion,
    st = :skew_taylor
)

# time step calculation
function get_dt(timestep_size, pc)
    if timestep_size[1] == :CFL
        dx_min = minimum(pc.p.h)
        for i = eachindex(pc), j = pc.p.neighbors[i]
            j == i && continue
            dx_min = min(dx_min, norm(pc.p.x[j] - pc.p.x[i]))
        end
        return timestep_size[2] * dx_min^2
    elseif timestep_size[1] == :CONST
        return timestep_size[2]
    end
end

# simulation name for heat equation
simulation_savename(simname, n, save) = save ? datadir(simname, "results_n=$n") : ""

# unique point cloud names
pointcloud_name(h) = datadir("pointclouds", "Square_h=$(h)_type=meshfree.csv")

# number of points in pc
N(h) = countlines(pointcloud_name(h)) - 1

# wrapper for loading point clouds
function load_pc(h; h_rescale_factor=1.0)
    pc = load_csv(pointcloud_name(h))
    rescale!(pc)
    pc.p.h[:] *= h_rescale_factor
    labelpoints!(pc, Val(:square))
    return pc
end

# generic solver for PDEs
function generate_results(
    testname, model, bcon, u;
    hs       = [0.2/2^k for k=0:5],
    ps_ddo   = (2, 4),
    averages = AVG_DEFAULT,
    ps_mls   = (2, 4),
    kwargs...
)
    println("Running ", basename(testname))

    # temporary directory for saving intermediate results to avoid recomputation
    tempsimdir = mkpath(joinpath(dirname(@__FILE__), ".temp"))

    results = DataFrame()

    for h = hs
        @show h

        ddo_runs = map(methods_ddo(ps_ddo, averages)) do method
            Threads.@spawn begin
                println("Running ", method.name, "...")
                result = run_single_simulation(
                    model, bcon, method.method, h, u; kwargs...
                )
                println("Finished ", method.name, ": ", result)
                (; zip((method.name, ), (result, ))...)
            end
        end

        mls_runs = map(methods_mls(ps_mls)) do method
            Threads.@spawn begin
                println("Running ", method.name, "...")
                result = run_single_simulation(
                    model, bcon, method.method, h, u;
                    gradient_order=method.gradient_order, kwargs...
                )
                println("Finished ", method.name, ": ", result)
                (; zip((method.name, ), (result, ))...)
            end
        end

        errors_ddo = fetch.(ddo_runs)
        errors_mls = fetch.(mls_runs)

        row = merge(errors_ddo..., errors_mls...)
        push!(results, (; h=h, N=N(h), row...))

        # write temporary results
        CSV.write(joinpath(tempsimdir, first(splitext(basename(testname)))*".csv"), results)
    end

    # remove temporary file
    rm(tempsimdir, recursive=true)

    return results
end

# specific solver for Poisson equation
function run_single_simulation(
    model::PoissonEquation, bcon, method, h, u;
    linear_system_solver = LinearSolver_bicgstabl(),
    gradient_order       = 2,
    h_rescale_factor     = 1.0,
)
    pc = load_pc(h; h_rescale_factor)
    solve!(pc, model, bcon; linear_system_solver, method, gradient_order)
    return li_error(pc.p.T, u.(pc.p))
end

# specific solver for heat equation
function run_single_simulation(
    model::HeatEquation, bcon, method, h, u;
    linear_system_solver = LinearSolver_bicgstabl(),
    time_start           = 0.0,
    time_end             = 1.0,
    timestep_size        = (:CFL, 0.7),
    timestep_method      = LinearizedImplicitEuler(),
    gradient_order       = 2,
    filename             = "",
    h_rescale_factor     = 1.0,
)
    pc = load_pc(h; h_rescale_factor)
    dt = get_dt(timestep_size, pc)
    df = solve!(
        pc, model, bcon;
        gradient_order,
        linear_system_solver,
        timestep_method,
        save_items=(U  = (p, t) -> u.(p, t),),
        integrations=(error = (p, t) -> li_error(p.T, p.U),),
        timespan=(time_start, time_end),
        Δt_max=dt,
        method=method,
        save_interval=1,
        filename=filename,
        verbose=false,
    )
    return maximum(df.error)
end
