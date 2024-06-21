abstract type AbstractBoundaryFlag{T} end
flag(::AbstractBoundaryFlag{T}) where {T} = T

struct Dirichlet <: AbstractBoundaryFlag{:dirichlet} end
struct Neumann{DD <: AbstractDiagonalDominance} <: AbstractBoundaryFlag{:neumann}
    order::Int
    diagonal_dominance::DD
end

struct ConservativeNeumann <: AbstractBoundaryFlag{:conservative_neumann} end

struct Jump <: AbstractBoundaryFlag{:jump} end
struct FluxJump{DD <: AbstractDiagonalDominance} <: AbstractBoundaryFlag{:fluxjump}
    order::Int
    diagonal_dominance::DD
end

abstract type AbstractBoundaryCondition end
struct BoundaryCondition{F, BC <: AbstractBoundaryFlag} <: AbstractBoundaryCondition
    f::F
    bc::BC
end

flag(bcon::BoundaryCondition) = flag(bcon.bc)
(bcon::BoundaryCondition)(args...; kwargs...) = bcon.f(args...; kwargs...)

BoundaryCondition(a::Number, bc::AbstractBoundaryFlag) = BoundaryCondition((args...; kwargs...) -> a, bc)
BoundaryCondition(f, bcon::BoundaryCondition)          = BoundaryCondition(f, bcon.bc)

# shortcuts for common boundary conditions
Mimic()                                                 = BoundaryCondition(0, Jump())
Dirichlet(f)                                            = BoundaryCondition(f, Dirichlet())
Neumann(f; order = 1, diagonal_dominance = OneDimensionalCorrection_Default())  = BoundaryCondition(f, Neumann(order, diagonal_dominance))
ConservativeNeumann(f) = BoundaryCondition(f, ConservativeNeumann())
Jump(f)                                                 = BoundaryCondition(f, Jump())
FluxJump(f; order = 1, diagonal_dominance = OneDimensionalCorrection_Default()) = BoundaryCondition(f, FluxJump(order, diagonal_dominance))


# assumes f = f(p, t) ==> fᵢⁿ = f(pᵢ, tⁿ)
function boundary_at_timelevel(bcons::NamedTuple, t::Number)
    return NamedTuple{keys(bcons)}(
        map(bcon -> BoundaryCondition(x -> bcon.f(x, t), bcon), values(bcons))
    )
end

function check_consistency_boundary(pc, boundaryconditions)
    passed = true
    for label = unique(pc.p.label)
        (label == "" || label == Symbol()) && continue
        haskey(boundaryconditions, String(label)) && continue
        haskey(boundaryconditions, Symbol(label)) && continue
        @warn "Boundary conditions need to be set for variable $var and boundary labeled $label."
        passed = false
    end
    !passed && error("Some boundary conditions have not been set!")
end

function extract_boundary(boundaryconditions::Dict, var::Symbol)
    return NamedTuple(
        (Symbol(key[2]), boundaryconditions[var, key[2]]) for key = keys(boundaryconditions)
    )
end

function assign_boundaryflag!(pc, bcons)
    for (label, bcon) = pairs(bcons)
        plist = findall(==(label), pc.p.label)
        pc.p.bnd_flag[plist] .= flag(bcon)
    end
end