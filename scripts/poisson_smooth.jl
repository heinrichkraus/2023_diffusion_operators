include("base.jl")

function run_poisson_smooth()
    # solution
    u(p) = prod(sinpi, p.x)
    Δu(p) = -2π^2 * u(p)
    ∇u(p) = π * SVector(cospi(p.x[1]) * sinpi(p.x[2]), sinpi(p.x[1]) * cospi(p.x[2]))

    # diffusivity
    λ(p) = exp(p.x[1] - p.x[2]^2)
    ∇λ(p) = λ(p) * SVector(1, -2p.x[2])

    # heat source
    q(p) = -(dot(∇u(p), ∇λ(p)) + λ(p) * Δu(p))

    save_testcase(@__FILE__, u, λ, 0.05)

    model = PoissonEquation(λ, q)
    bcon  = (left=Dirichlet(u), right=Dirichlet(u), top=Dirichlet(u), bottom=Dirichlet(u))

    results = generate_results(@__FILE__, model, bcon, u; h_rescale_factor=1.5)
    save_data(@__FILE__, results)

end

run_poisson_smooth()
