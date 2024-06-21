include("base.jl")

function run_heat_discontinuous()
    f(t)  = exp(-4t)
    df(t) = -4 * f(t)
    g(p)  = prod(sinpi, p.x)
    Δg(p) = -2π^2 * g(p)

    λ(p, t) = g(p) >= 3/4 ? 1e8 : 1e0
    u(p, t) = f(t) * ((g(p) - 3/4) / λ(p, t) + 3/4)
    q(p, t) = df(t) * g(p) - f(t) * Δg(p)

    model = HeatEquation((p, t) -> 1, λ, q, p -> u(p, 0))
    bcon  = (left=Dirichlet(u), right=Dirichlet(u), top=Dirichlet(u), bottom=Dirichlet(u))

    results = generate_results(
        @__FILE__, model, bcon, u;
        ps_ddo=(2, ),
        averages=(hm=:harmonic_mean, ),
        ps_mls=(2, ),
        timestep_size=(:CFL, 0.7),
        timestep_method=LinearizedTrapezoidalRule(),
        linear_system_solver=LinearSolver_bicgstabl(abstol=1e-12, reltol=1e-12),
        hs=[0.2/2^k for k = 0:3],
        h_rescale_factor=1.0,
    )

    save_data(@__FILE__, results)

end

run_heat_discontinuous()
