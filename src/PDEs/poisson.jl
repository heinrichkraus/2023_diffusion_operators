"""
Sets up poisson equation \n
    -∇⋅(λ∇T) = q.
Dirichlet and neumann boundary conditions can be used.
"""
struct PoissonEquation{Λ, Q} <: AbstractPartialDifferentialEquation
    λ::Λ
    q::Q
end


function prepare_timestep!(
    pc::PointCloud;
    compute_triangulation = true,
    checks = false, αtol = deg2rad(15),
    compute_interfaces = true,
    compute_neighborhoods = true,
    compute_sparsitypattern = true,
    compute_default_operators = true,
    kwargs...
)
    if compute_neighborhoods
        neighbors!(pc)
    end

    if compute_triangulation
        delaunay!(pc; checks, αtol)
        volumes!(pc; kwargs...)
    end

    if compute_interfaces && isseparated(pc)
        interfaces!(pc)
        neighbors!(pc)
    end

    if compute_sparsitypattern
        pc.sparsity = COOPattern(pc.p.neighbors)
    end

    if compute_default_operators
        default_operators!(pc; kwargs...)
    end
end

function setvars!(pc::PointCloud, model::PoissonEquation)
    pc.p.λ .= model.λ.(particles(pc))
    pc.p.q .= model.q.(particles(pc))
end

function solve!(
    pc::PointCloud{2},
    model::PoissonEquation,
    boundaryconditions;
    method = DiffusionSystem(
        Smoothing(),
        DiffusionOperator_Single(DivEtaGrad_ScaledLaplacian(:harmonic_mean, Laplace_WLSQ()))
    ),
    prepared_timestep = false,
    linear_system_solver = LinearSolver_bicgstabl(),
    kwargs...
)
    check_consistency_boundary(pc, boundaryconditions)

    # bfunT = extract_boundary(boundaryconditions, :T)

    prepared_timestep || prepare_timestep!(pc; kwargs...)

    setvars!(pc, model)

    M = zero(pc.sparsity)
    B = zero(pc.sparsity)

    rm = zeros(length(pc))
    rb = zeros(length(pc))

    assemble_diffusion_system!(M, rm, B, rb, model.λ, model.q, boundaryconditions, method, pc; kwargs...)

    A = B - M
    r = rm + rb

    solve_linear_system!(pc.p.T, A, r, linear_system_solver)
end
