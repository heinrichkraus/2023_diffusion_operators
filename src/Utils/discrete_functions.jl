struct DiscreteFunction{X <: AbstractVector, Y <: AbstractVector}
    x::X
    y::Y
end


### Evaluate discrete functions
### There should be an option for higher order interpolations!
function (df::DiscreteFunction{X, Y})(x::Real) where {X <: AbstractVector{<:Real}, Y}
    x <= df.x[1]   && return df.y[1]
    x >= df.x[end] && return df.y[end]

    i::Int = findfirst(i -> df.x[i] <= x < df.x[i+1], eachindex(df.x))

    x == df.x[i] && return df.y[i]

    dx = df.x[i+1] - df.x[i]
    dy = df.y[i+1] - df.y[i]

    lambda = (x - df.x[i]) / dx

    return df.y[i] + lambda * dy
end

### Inverse function
function inverse(df::DiscreteFunction{X, Y}) where {X <: AbstractVector{<:Real}, Y <: AbstractVector{<:Real}}
    (issorted(df.y) || issorted(df.y, rev = true)) || error("Function is not bijective!")
    return DiscreteFunction(df.y, df.x)
end

function inverse(df::DiscreteFunction{X, Y}, y::Real) where {X <: AbstractVector{<:Real}, Y <: AbstractVector{<:Real}}
    df_inv = DiscreteFunction(df.y, df.x)
    return df_inv(y)
end

### Derivatives
function differentiate(f, x::AbstractVector{<:Real})
    y = f.(x)

    n  = length(x)
    dy = similar(y)

    A = zeros(3, 3)
    b = zeros(3)
    w = zeros(3)

    A[1, 1:3] .= (1, 1, 1)
    b[2]       = 1

    # somewhat dirty and should be punished...
    for i = eachindex(x)
        I = if i == 1
            1:3
        elseif i == n
            n-2:n
        else
            i-1:i+1
        end

        X = view(x, I) .- x[i]

        A[2,:] .= X
        A[3,:] .= X.^2
        w[:]   .= A \ b

        dy[i] = sum(w .* view(y, I))
    end

    return DiscreteFunction(x, dy)
end

function differentiate(df::DiscreteFunction{X, Y}) where {X <: AbstractVector{<:Real}, Y}
    return differentiate(df, df.x)
end

### Antiderivatives and numerical quadrature
function antiderivative(f, x::AbstractVector{<:Real}; integration = GaussLegendre(1), offset = 0)
    y    = Vector{typeof(Float64.(f(x[1])))}(undef, length(x))
    y[1] = 0
    for i = 2:length(x)
        y[i] = y[i-1] + integrate(f, x[i-1], x[i], integration)
    end
    y .+= offset

    return DiscreteFunction(x, y)
end

function antiderivative(f::DiscreteFunction{X, Y}; kwargs...) where {X <: AbstractVector{<:Real}, Y}
    return antiderivative(f, f.x; kwargs...)
end

## Numerical quadrature, should probably be somewhere else
abstract type AbstractQuadratureFormula end
struct TrapezoidalRule <: AbstractQuadratureFormula end
struct GaussLegendre{K} <: AbstractQuadratureFormula end
GaussLegendre(k::Integer) = GaussLegendre{k}()

function integrate(f, a, b, integrator::AbstractQuadratureFormula)
    weights, points = integration_weights_points(a, b, integrator)
    return sum(@. weights * f(points))
end

function integration_weights_points(a, b, ::GaussLegendre{1})
    return ([b - a], [(a + b) / 2])
end

function integration_weights_points(a, b, ::GaussLegendre{2})
    dx = (b - a) / 2
    xm = (a + b) / 2

    w = [1, 1] * dx
    p = xm .+ (dx ./ [-sqrt(3), sqrt(3)])

    return (w, p)
end

function integration_weights_points(a, b, ::GaussLegendre{3})
    dx = (b - a) / 2
    xm = (a + b) / 2

    w = [5/9, 8/9, 5/9] * dx
    p = xm .+ (dx .* [-sqrt(3/5), 0, sqrt(3/5)])

    return (w, p)
end

function integration_weights_points(a, b, ::TrapezoidalRule)
    dx = (b - a) / 2

    w = [1, 1] * dx
    p = [a, b]

    return (w, p)
end