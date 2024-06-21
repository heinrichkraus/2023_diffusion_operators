abstract type AbstractDiagonalDominance end

abstract type AbstractDiagonalDominance_On  <: AbstractDiagonalDominance end
abstract type AbstractDiagonalDominance_Off <: AbstractDiagonalDominance end

abstract type AbstractOneDimensionalCorrection <: AbstractDiagonalDominance_On end

struct DD_On  <: AbstractDiagonalDominance_On  end
struct DD_Off <: AbstractDiagonalDominance_Off end

"""
Minimizes\n
    φ(α) = v * (∑ⱼᵥᵢ (cⱼ + α hⱼ)^2 / (cᵢ + α hᵢ)^2)
for diagonal dominance
"""
struct OneDimensionalCorrection_Default <: AbstractOneDimensionalCorrection end

function α_opt(c, h, i, ::OneDimensionalCorrection_Default)
    hh = dot(h,h)
    hc = dot(h,c)
    cc = dot(c,c)

    return (c[i] * hc - h[i] * cc) / (h[i] * hc  - c[i] * hh)

end

"""
Minimizes\n
φ(α) = v * (∑ⱼᵥᵢ (cⱼ + α hⱼ)^2 / (cᵢ + α hᵢ)^2) + ∑ⱼ w max(0, -cⱼ + α hⱼ)^k₁ + w max(0, (cᵢ + α hᵢ))^k₁ + σ max(0, ∑ⱼ |cⱼ + α hⱼ| - |cᵢ + α hᵢ| )^k₂
for diagonal dominance
    """
Base.@kwdef struct OneDimensionalCorrection_Penalty <: AbstractOneDimensionalCorrection
    v::Float64  = 0.0
    w::Float64  = 1.0
    σ::Float64  = 1.0
    k1::Float64 = 1.0
    k2::Float64 = 1.0
    n_iter::Int = 50
end

function α_opt(c, h, i, dd::OneDimensionalCorrection_Penalty)
    ϕ(α) = opt_penaltyfun(α, c, h, i, dd)
    return α_opt(c, h, i, OneDimensionalCorrection(; ϕ))
end

"""
Returns optimization function for maximal diagonal dominance with respect to index i of the shape \n
    φ(α) =  v * (∑ⱼᵥᵢ (cⱼ + α hⱼ)^2 / (cᵢ + α hᵢ)^2) + ∑ⱼ w max(0, -cⱼ + α hⱼ)^k₁ + w max(0, (cᵢ + α hᵢ))^k₁ + σ max(0, ∑ⱼ |cⱼ + α hⱼ| - |cᵢ + α hᵢ| )^k₂
"""
function opt_penaltyfun(α, c, h, i, dd::OneDimensionalCorrection_Penalty)
    s1 = zero(Float64)
    s2 = zero(Float64)
    s3 = zero(Float64)

    for j = eachindex(c)
        aj = c[j] + α * h[j]
        s1 += aj^2
        if j != i
            # off-diagonal elements
            s2 += max(0.0, -aj)^dd.k1
            s3 += abs(aj)
        else
            # diagonal elements
            s2 += max(0.0, aj)^dd.k1
            s3 -= abs(aj)
        end
    end

    # finalizing optimization function
    s1 *= dd.v / (c[i] + α * h[i])^2
    s2 *= dd.w
    s3  = dd.σ * max(0.0, s3)^dd.k2

    if dd.v != 0.0
        return s1 + s2 + s3
    else
        return s2 + s3
    end
end

function α_OPTIM(f, α_0::Number, optimizer, options; verbose=false)
    out = Optim.optimize(f, [α_0], optimizer, options)
    α   = Optim.converged(out) ? first(out.minimizer) : α_0
    verbose && display(out)
    verbose && display(α)
    return α
end

function α_OPTIM(f, α_0::AbstractVector, optimizer, options; verbose=false)
    out = Optim.optimize(f, α_0, optimizer, options)
    α   = Optim.converged(out) ? out.minimizer : α_0
    verbose && display(out)
    verbose && display(α)
    return α
end

Base.@kwdef struct OneDimensionalCorrection{Φ, OPT} <: AbstractOneDimensionalCorrection
    ϕ::Φ
    optimizer::OPT = (f, α_0) -> α_OPTIM(f, α_0, Optim.LBFGS(), Optim.Options(); verbose=false)
end

function α_opt(c, h, i, dd::OneDimensionalCorrection)
    fun(α) = dd.ϕ(c + first(α) * h)
    α = α_opt(c, h, i, OneDimensionalCorrection_Default())
    return dd.optimizer(fun, α)
end

abstract type AbstractMultiDimensionalCorrection <: AbstractDiagonalDominance end
Base.@kwdef struct MultiDimensionalCorrection{Φ, OPT} <: AbstractMultiDimensionalCorrection
    ϕ::Φ
    optimizer::OPT = (f, α_0) -> α_OPTIM(f, α_0, Optim.LBFGS(), Optim.Options(); verbose=false)
end

function α_opt(c, H, i, dd::MultiDimensionalCorrection)
    fun(α) = dd.ϕ(c + H * α)
    α = [α_opt(c, h, i, OneDimensionalCorrection_Default()) for h = eachcol(H)]
    return dd.optimizer(fun, α)
end

struct MultiDimensionalCorrection_Standard <: AbstractMultiDimensionalCorrection end