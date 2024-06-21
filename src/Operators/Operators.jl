include("LocalOperators/abstract.jl")
include("LocalOperators/approximation.jl")
include("LocalOperators/directionalderivative.jl")
include("LocalOperators/laplace.jl")
include("LocalOperators/smoothing.jl")
include("LocalOperators/div_eta_grad.jl")
include("LocalOperators/neumann.jl")

include("LocalOperators/boundaryconditions.jl")

include("GlobalOperators/diffusion_system.jl")

function default_operators!(
    pc::PointCloud;
    gradient_order=2,
    laplace_order=2,
    neumann_order=1,
    kwargs...
)
    # names = coordnames(Val(dim))

    for i = eachindex(pc)
        pc.p.c_gradient[i] = isboundary(pc.p[i]) ?
            gradient_row(pc.p.x[i], pc.p.h[i], pc.p.neighbors[i], 1, pc, DD_Off()) :
            gradient_row(pc.p.x[i], pc.p.h[i], pc.p.neighbors[i], gradient_order, pc, DD_Off())
        pc.p.c_laplace[i]  = isboundary(pc.p[i]) ?
            directional_derivative_row(pc.p.n[i], pc.p.x[i], pc.p.h[i], pc.p.neighbors[i], neumann_order, pc, OneDimensionalCorrection_Default()) :
            laplace_row(i, pc, Laplace_WLSQ(order=laplace_order))
    end

end
