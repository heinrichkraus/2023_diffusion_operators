
Base.@kwdef mutable struct Particle{dim, Tv <: AbstractFloat}
    # coordinates
    x::SVector{dim, Tv} = zero(SVector{dim, Tv})
    n::SVector{dim, Tv} = zero(SVector{dim, Tv})

    # neighborhoods
    h::Tv = zero(Tv) #?
    neighbors::Vector{Int} = Int[] #? different data structure

    # boundary
    label::Symbol = Symbol()
    ident::Symbol = :none
    bnd_flag::Symbol = Symbol() #? only needed in certain context

    # triangulation
    simplices::Vector{Int} = Int[] #? different data structure
    dV::Tv = zero(Tv)
    dS::Tv = zero(Tv)

    # domains
    domain::Int = 0     # ideally introduce a notion of multiple point clouds where domain is not necessary
    partner::Int = 0    # here, I would need to store the partner and opposite point cloud index. each point would also need to get a unique global identifier for its position in the linear system
    # generally, this would simplify domain decomposition computations but add a lot of work

    # operators
    c_gradient::Matrix{Tv} = Matrix{Tv}(undef, 0, 0)
    c_laplace::Vector{Tv}  = Tv[]

    # !Ideally the information below is not stored here but in the corresponding methods.
    # temporary data
    eta::Tv = one(Tv)
    mue::Tv = one(Tv)

    # physical properties
    c::Tv = one(Tv)            # (apparent) specific heat capacity
    T::Tv = zero(Tv)            # temperature
    λ::Tv = one(Tv)            # heat conductivity
    q::Tv = zero(Tv)            # (volumetric) heat source

    # old physical properties
    c0::Tv = one(Tv)            # (apparent) specific heat capacity
    T0::Tv = zero(Tv)            # temperature
    λ0::Tv = one(Tv)            # heat conductivity
    q0::Tv = zero(Tv)            # (volumetric) heat source

    # operator analysis
    dd_check::Bool = false
    dd_error::Float64 = 0.0
end

Particle(x::AbstractVector; kwargs...) = Particle{length(x), eltype(x)}(; x, kwargs...)

Base.zero(::Type{Particle{dim, Tv}}) where {dim, Tv} = Particle(zero(SVector{dim, Tv}))

dimension(::Particle{dim, Tv}) where {dim, Tv} = dim
precision(::Particle{dim, Tv}) where {dim, Tv} = Tv

isboundary(p::Particle) = p.ident ∈ (:free_surface, :wall)
isghost(p) = p.ident == :ghost
isgeometry(p) = p.ident == :geometry

@inline coordnames(::Particle{dim, Tv})  where {dim, Tv} = coordnames(Val(dim))
@inline normalnames(::Particle{dim, Tv}) where {dim, Tv} = normalnames(Val(dim))
