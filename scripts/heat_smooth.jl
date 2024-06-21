include("base.jl")

function run_heat_smooth()
    # u(p, t) = f(t) * g(p)
    #f
    f(t)  = exp(-4t)
    df(t) = -4 * f(t)

    #g
    g(p)  = prod(sinpi, p.x)
    ∇g(p) = π * SVector(cospi(p.x[1]) * sinpi(p.x[2]), sinpi(p.x[1]) * cospi(p.x[2]))
    Δg(p) = -2π^2 * g(p)

    #u
    u(p, t) = f(t) * g(p)
    ∇u(p, t) = f(t) * ∇g(p)
    Δu(p, t) = f(t) * Δg(p)

    #λ
    λ(p)  = exp(p.x[1] - p.x[2]^2)
    ∇λ(p) = exp(p.x[1] - p.x[2]^2) * SVector(1, -2p.x[2])
    λ(p, t) = λ(p)

    #q
    q(p, t) = df(t) * g(p) - dot(∇λ(p), ∇u(p, t)) - λ(p) * Δu(p, t)

    model = HeatEquation((p, t) -> 1, λ, q, p -> u(p, 0))
    bcon  = (left=Dirichlet(u), right=Dirichlet(u), top=Dirichlet(u), bottom=Dirichlet(u))

    results = generate_results(
        @__FILE__, model, bcon, u;
        linear_system_solver = LinearSolver_bicgstabl(),
        averages=(am=:arithmetic_mean, hm=:harmonic_mean),
        timestep_size=(:CFL, 0.7),
        timestep_method=LinearizedTrapezoidalRule(),
        hs=[0.2/2^k for k = 0:3],
        h_rescale_factor=1.5
    )

    save_data(@__FILE__, results)

end

run_heat_smooth()
