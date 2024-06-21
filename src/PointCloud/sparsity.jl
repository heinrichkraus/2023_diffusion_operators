abstract type AbstractSparsityPattern end

struct COOPattern <: AbstractSparsityPattern
    I::Vector{Int}
    J::Vector{Int}
    n::Int
    m::Int
end

function Base.zero(pattern::COOPattern)
    return sparse(pattern, zeros(length(pattern.I)))
end

function SparseArrays.sparse(pattern::COOPattern, vals)
    length(vals) != length(pattern.I) && error("sparse: length(vals) != pattern.nvals")
    return sparse(pattern.I, pattern.J, vals, pattern.n, pattern.m)
end

function COOPattern(neighborlist::Vector{Vector{Int}})
    n = length(neighborlist)
    J = reduce(vcat, neighborlist)
    I = similar(J)
    cnt = 0
    for i = eachindex(neighborlist)
        N = length(neighborlist[i])
        I[cnt + 1 : cnt + N] .= i
        cnt += N
    end
    return COOPattern(I, J, n, n)
end