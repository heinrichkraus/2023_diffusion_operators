abstract type AbstractDiffusionScheme end

struct DiffusionOperator_Single <: AbstractDiffusionScheme
    method
end

struct DiffusionOperator_CheckDiagonalDominance{
    MP1 <: AbstractDiffusionOperator,
    MP2 <: AbstractDiffusionOperator
} <: AbstractDiffusionScheme
    primary::MP1
    secondary::MP2
    extend_to_neighbors::Bool
end

DiffusionOperator_CheckDiagonalDominance(; primary, secondary, extend_to_neighbors = false) = DiffusionOperator_CheckDiagonalDominance(primary, secondary, extend_to_neighbors)

struct DiffusionSystem{SO <: Smoothing, DO <: AbstractDiffusionScheme}
    smoothing::SO
    operator::DO
end

DiffusionSystem(; operator, smoothing = Smoothing()) = DiffusionSystem(smoothing, operator)

function set_discrete_diffusion!(
    D, q, plist, pc, method
)
    for i = plist
        D[i, pc] = div_eta_grad_row(i, pc, method)
        q[i]     = pc.p.q[i]
    end
end

function assemble_diffusion_system!(
    D, rd,                  # discrete differential operator and corresponding rhs
    B, rb,                  # boundary conditions with rhs
    λ,                      # heat conductivity
    q,                      # (volumetric) heat source
    bcons,                  # boundary conditions
    diffusion::DiffusionSystem, # just use one single method
    pc::PointCloud;
    kwargs...
)

    pc.p.eta .= λ.(pc.p)
    pc.p.mue .= log.(pc.p.eta)

    smooth!(pc.p.eta, pc, diffusion.smoothing)
    smooth!(pc.p.mue, pc, diffusion.smoothing)

    plist = findall(!isboundary, pc.p)

    rd[plist] = q.(pc.p[plist])
    set_diffusion_system!(D, diffusion.operator, plist, pc; kwargs...)

    map((label, bcon) -> parse_boundary_condition_base!(B, rb, label, bcon, pc), keys(bcons), values(bcons))

end

function set_diffusion_system!(
    D,
    method::DiffusionOperator_Single,
    plist,
    pc;
    kwargs...
)
    for i = plist
        D[i, pc] = div_eta_grad_row(i, pc, method.method; kwargs...)
    end

end

function set_diffusion_system!(
    D,
    method::DiffusionOperator_CheckDiagonalDominance,
    plist,
    pc;
    kwargs...
)
    recompute = zeros(Bool, length(pc))

    for i = plist
        c = div_eta_grad_row(i, pc, method.primary)
        D[i, pc] = c

        rowsum = sum(abs, (c[2:end] ./ abs(c[1]))) - 1
        if rowsum > 1e-12 || c[1] >= 0
            recompute[i] = true
            if method.extend_to_neighbors
                recompute[pc.p.neighbors[i]] .= true
            end
        end
    end

    plist2 = intersect(plist, findall(recompute))

    for i = plist2
        D[i, pc] = div_eta_grad_row(i, pc, method.secondary)
    end
end

# possible operator types:
# single operator: apply same scheme to all (inner) points
# hybrid operator: apply different schemes to different sets of points, maybe even with propagation onto the boundary (e.g. conservative Neumann boundaries)
#       - could a priori define where which method is used (e.g. looking for jump discontinuities, special locations, ...)
#       - could define a check that has to be run during the assembly process (stability check, e.g. diagonal dominance) in order to identify critical regions and escape to alternative method
# mixed operator: determine a priori sets of points (e.g. with a condition) and apply differential operators on them
# note: smoothing can be done for any of these methods!

# general workflow:
# 1. smooth eta / mue
# 2. discretize ∇⋅(η∇u) and q for inner points
# 3. discretize boundary conditions (... could be used somewhere else, also... maybe?)
