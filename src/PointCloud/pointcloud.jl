mutable struct PointCloud{dim, Tv <: AbstractFloat, Particles <: AbstractVector{Particle{dim, Tv}}}
    p::Particles
    debug::DataFrame
    cells::Vector{<:AbstractVector}
    sparsity::COOPattern
end

function PointCloud(
    particles::AbstractVector{Particle{dim, Tv}};
    debug = DataFrame(),
    triangulation = zeros(SVector{dim +1, Int}, 0),
    sparsity = COOPattern(Int[], Int[], 0, 0),
) where {dim, Tv}
    return PointCloud(particles, debug, triangulation, sparsity)
end

# Base.show(io::IO, pc::PointCloud) = print(io, PrettyTables.pretty_table(pc.p))

@inline dimension(::PointCloud{dim, Tv, Data}) where {dim, Tv, Data} = dim
@inline precision(::PointCloud{dim, Tv, Data}) where {dim, Tv, Data} = Tv

particles(pc::PointCloud) = pc.p

xs(pc::PointCloud) = view(tocolmatrix(pc.p.x), 1, :)
ys(pc::PointCloud) = view(tocolmatrix(pc.p.x), 2, :)
zs(pc::PointCloud) = view(tocolmatrix(pc.p.x), 3, :)

nxs(pc::PointCloud) = view(tocolmatrix(pc.p.n), 1, :)
nys(pc::PointCloud) = view(tocolmatrix(pc.p.n), 2, :)
nzs(pc::PointCloud) = view(tocolmatrix(pc.p.n), 3, :)

# @inline Base.propertynames(pc::PointCloud{dim, Tv, Data}) where {dim, Tv, Data} =
#         tuple(
#             propertynames(getfield(pc, :particles))...,
#             fieldnames(PointCloud)...
#         )

# @inline function Base.getproperty(pc::PointCloud{dim, Tv, Data}, f::Symbol) where {dim, Tv, Data}
#     if hasfield(PointCloud, f)
#         return getfield(pc, f)
#     elseif hasproperty(getfield(pc, :particles), f)
#         return getproperty(getfield(pc, :particles), f)
#     else
#         error("Point cloud has no property $f.")
#     end
# end

# @inline Base.propertynames(pc::PointCloud{dim, Tv, Data}) where {dim, Tv, Data} =
#         tuple(
#             coordnames(Val(dim))...,
#             normalnames(Val(dim))...,
#             propertynames(getfield(pc, :particles))...,
#             fieldnames(PointCloud)...
#         )

# @inline function Base.getproperty(pc::PointCloud{dim, Tv, Data}, f::Symbol) where {dim, Tv, Data}
#     if hasfield(PointCloud, f)
#         return getfield(pc, f)
#     elseif hasproperty(getfield(pc, :particles), f)
#         return getproperty(getfield(pc, :particles), f)
#     elseif coordposition(f, Val(dim)) > 0
#         return getcoordinates(f, pc, Val(dim))
#     elseif normalposition(f, Val(dim)) > 0
#         return getnormals(f, pc, Val(dim))
#     else
#         error("Point cloud has no property $f.")
#     end
# end

# # extract coordinates
# getcoordinates(f, pc, ::Val{dim}) where dim = view(tocolmatrix(getproperty(pc, :point)),  coordposition(f, Val(dim)), :)
# getnormals(f, pc, ::Val{dim})     where dim = view(tocolmatrix(getproperty(pc, :normal)), normalposition(f, Val(dim)), :)

tocolmatrix(v::Vector{SVector{dim, T}}) where {dim, T} = reshape(reinterpret(T, v), dim, :)
torowmatrix(v::Vector{SVector{dim, T}}) where {dim, T} = transpose(reshape(reinterpret(T, v), dim, :))
tovecvec(m::Matrix) = Array(vec(reinterpret(SVector{size(m,1), eltype(m)}, m)))

# indexing
@inline Base.length(pc::PointCloud)    = length(pc.p)
@inline Base.eachindex(pc::PointCloud) = Base.OneTo(length(pc))
