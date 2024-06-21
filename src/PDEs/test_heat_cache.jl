using LinearAlgebra, SparseArrays, BenchmarkTools
struct Cache{TM <: AbstractMatrix, TV <: AbstractVector}
    A::TM
    B::TM
    C::TM
    a::TV
    b::TV
    c::TV
end

n = 100_000
neighborlist = [i-10:i+10 for i = 1:n]
neighborlist = map(nbrs -> max(nbrs[1], 1):min(nbrs[end], n), neighborlist)
isbnd = rand(Bool, n)

I = vcat(map(i -> i * ones(Int, length(neighborlist[i])), 1:n)...)
J = vcat(neighborlist...)
# VA = map((i, j) -> (i==j) * float(isbnd[i]), I, J)
VA = randn(length(I))
VB = randn(length(I))

A = sparse(I, J, VA)
B = sparse(I, J, VB)
C = similar(A)

mc = Cache(A, B, C, rand(n), rand(n), zeros(n))

function set_cs!(mc)
    (; A, B, C, a, b, c) = mc
    @. C = B - A
    mul!(c, A, a)
    mul!(c, B, b, 1, 1)
    # mul!(c, C, c)
    mul!(a, C, c)
    c .= a
    # @. c = $(A * a)
    # @. c += $(B * b)
    # @. c = $(C * c)
end

function SPM_add!(R, A, B)
    @. R.nzval = A.nzval + B.nzval
end

function SPM_add2!(R, A, B)
    @. R = A + B
end

@btime set_cs!($mc);
@btime SPM_add!($C, $A, $B);
@btime SPM_add2!($C, $A, $B);

@show Base.summarysize(A);
@show Base.summarysize(VA) + Base.summarysize(I) + Base.summarysize(J);