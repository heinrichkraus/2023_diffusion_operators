"""
Sets up heat equation \n
    c ∂ₜT = ∇⋅(λ∇T) + q.
Dirichlet and neumann boundary conditions can be used.
"""
struct HeatEquation{C, Λ, Q, Τ} <: AbstractPartialDifferentialEquation
    c::C
    λ::Λ
    q::Q
    T::Τ
end

function HeatEquation(; heat_capacity, heat_conductivity, heat_source, initial_temperature)
    return HeatEquation(heat_capacity, heat_conductivity, heat_source, initial_temperature)
end

Base.@kwdef mutable struct TimeStepData
    t::Float64        = 0
    Δt::Float64       = 0
    iteration::UInt32 = 0
end

function setvars!(X, t, model::HeatEquation)
    for (i, x) = enumerate(X)
        X.c[i] = model.c(x, t)
        X.λ[i] = model.λ(x, t)
        X.q[i] = model.q(x, t)
    end
end

function initialize_temperature!(X, model::HeatEquation)
   X.T  .= model.T.(X)
   X.T0 .= model.T.(X)
end

timestep_name(filename, nb_timestep) = filename * "_timestep_$nb_timestep"

function eyeinner(pc::PointCloud)
    I = zero(pc.sparsity)
    plist = findall(==(Symbol()), pc.p.label)
    for i = plist
        I[i,i] = 1
    end
    return I
end

function cycle_data!(X, ::HeatEquation)
    for i = eachindex(X)
        X.T0[i] = X.T[i]
        X.c0[i] = X.c[i]
        X.λ0[i] = X.λ[i]
        X.q0[i] = X.q[i]
    end
end

function eyeinner2(pc::PointCloud)
    return Diagonal(pc.p.label .== Symbol())
end

struct HeatEquationCache{MType, VType}
    A::MType    # linear system matrix
    r::VType    # linear system rhs
    M::MType    # diffusion matrix
    rm::VType   # diffusion RHS (heat source)
    B::MType    # boundary matrix
    rb::VType   # boundary RHS
    I::MType    # interior points δ_ij
    dT::VType   # temperature displacement
end

function HeatEquationCache(sparsity::COOPattern, N::Integer)
    return HeatEquationCache(
        zero(sparsity), zeros(N),
        zero(sparsity), zeros(N),
        zero(sparsity), zeros(N),
        zero(sparsity), zeros(N)
    )
end

function solve!(
    pc::PointCloud, model::HeatEquation, bcon;
    method = DiffusionSystem(
        Smoothing(w -> exp(-3w), 1),
        DiffusionOperator_Single(
            DivEtaGrad_WLSQ(2, true, OneDimensionalCorrection_Default())
        )
    ),
    prepared_timestep = false,
    timespan,
    Δt_max::Number,
    filename = "",
    save_interval = 1,
    nb_maxiter = typemax(UInt32),
    integrations = NamedTuple(),
    save_items   = NamedTuple(),
    preprocessing = (),
    postprocessing = (),
    correction_algorithm = TemperatureCorrection_Off(),
    timestep_method = LinearizedImplicitEuler(),
    linear_system_solver = LinearSolver_bicgstabl(),
    verbose=false,
    kwargs...
)
    prepared_timestep || prepare_timestep!(pc; kwargs...)

    assign_boundaryflag!(pc, bcon)
    mcache = HeatEquationCache(pc.sparsity, length(pc))

    mcache.I .= eyeinner(pc)

    write_output = !isempty(filename)

    verbose && @show write_output

    filepath = abspath(filename)
    fpath, fname = splitdir(filepath)
    mkpath(fpath)


    # initial configuration
    tstart, tend = timespan
    tsd = TimeStepData(t = tstart, Δt = Δt_max)
    initialize_temperature!(pc.p, model)
    setvars!(pc.p, tsd.t, model)

    saves = Table(pc, save_items)

    prepare_save_items!(saves, tsd.t, save_items)

    if write_output
        paraview_output = paraview_collection(filepath)
        paraview_output[tsd.t] = vtk_file(timestep_name(filepath, 0), pc, saves)
    end

    df_integrations = initialize_integrations(
        saves, tsd.t, integrations;
        t = tsd.t, Δt = tsd.Δt, iteration = tsd.iteration,
    )

    # time iteration
    while tsd.t < tend && tsd.iteration <= nb_maxiter
        # possibly calculate Δt
        tsd.Δt = min(tsd.Δt, tend - tsd.t)

        # prepare data (needed to have correct values for c0, λ0, q0)
        setvars!(pc.p, tsd.t, model)
        cycle_data!(particles(pc), model)

        for f = preprocessing
            f(saves, tsd.t, pc)
        end

        perform_timestep!(pc, tsd.t, tsd.Δt, model, bcon, mcache, timestep_method, method, linear_system_solver; kwargs...)
        correct_dT!(mcache.dT, correction_algorithm, tsd, model, pc)

        for i = eachindex(pc)
            pc.p.T[i] = pc.p.T0[i] + mcache.dT[i]
        end

        for f = postprocessing
            f(saves, tsd.t, pc)
        end

        # update important data
        tsd.t += tsd.Δt

        # save
        tsd.iteration += 1
        prepare_save_items!(saves, tsd.t, save_items)
        evaluate_integrations!(
            df_integrations, saves, tsd.t, integrations;
            t = tsd.t, Δt = tsd.Δt, iteration = tsd.iteration,
        )

        if mod(tsd.iteration, save_interval) == 0
            if write_output
                paraview_output[tsd.t] = vtk_file(
                    timestep_name(filepath, div(tsd.iteration, save_interval)), pc, saves
                )
            end

        end

        verbose && println("t = $(tsd.t), iteration = $(tsd.iteration)")
    end

    write_output && vtk_save(paraview_output)

    return df_integrations
end
