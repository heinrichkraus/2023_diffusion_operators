"""
Computes Lp norm given by
    ‖u‖ₚ = (∫_Ω |u|ᵖ dV)^(1/p)
for a function defined on Ω.
"""
function LinearAlgebra.norm(u::AbstractVector{Tv}, pc::PointCloud{dim, Tv, Data}, p::Real = 2) where {dim, Tv, Data}
    length(pc) != length(u) && error("Length of u and point cloud do not match.")

    out = zero(Tv)
    if isinf(p)
        out = norm(u, Inf)
    else
        for i = eachindex(pc)
            out += pc.p.dV[i] * abs(u[i])^p
        end
        out = out^(1/p)
    end
    return out
end

function integrate(u::AbstractVector, pc::PointCloud)
    res = 0.0
    for i = eachindex(pc)
        res += pc.p.dV[i] * u[i]
    end
    return res
end