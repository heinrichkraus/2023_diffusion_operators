include("base.jl")

function run_poisson_discontinuous()
    # base function
    H = 3/4

    uu(p) = prod(sinpi, p.x)
    q(p)  = 2π^2 * uu(p) #q = -2Δuu

    # diffusivity
    λ(p) = uu(p) > H ? 1e8 : 1.0

    # solution
    u(p) = (uu(p) - H) / λ(p) + H

    save_testcase(@__FILE__, u, λ, 0.05)

    model = PoissonEquation(λ, q)
    bcon  = (left=Dirichlet(u), right=Dirichlet(u), top=Dirichlet(u), bottom=Dirichlet(u))

    results = generate_results(
        @__FILE__, model, bcon, u;
        linear_system_solver=LinearSolver_bicgstabl(abstol=1e-12, reltol=1e-12),
        ps_ddo = (2, ),
        ps_mls = (2, ),
        h_rescale_factor=1.0,
    )
    save_data(@__FILE__, results)
end

run_poisson_discontinuous()
