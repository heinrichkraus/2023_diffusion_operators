"""
Returns solution, its derivatives and right-hand side of div_eta_grad-problem where f is taken as a basis function for the solution \n
    u(x) = (f(x) - H) / η(x) + H.
H is the function value of the contour line where f is cut \n

and η returns ηL or ηR depending on the position of the contour line.
"""
function contour(etaL, etaR, H, f, ∇f, Δf)
    function η(x)
        if f(x) <= H
            return etaL
        else
            return etaR
        end
    end

    u(x)  = (f(x) - H) / η(x) + H
    ∇u(x) = ∇f(x) / η(x)

    return η, u, ∇u, Δf
end

"""
    f(x) = exp(-α*‖x‖²)
"""
function contour_radial_exp(etaL, etaR, r; α = 1.0)
    println("α = $α")
    f(x)  = exp(-α * norm(x)^2)
    ∇f(x) = -2α * f(x) * x
    Δf(x) = -2α * (length(x) - 2α * norm(x)^2) * f(x)
    return contour(etaL, etaR, exp(-α * r^2), f, ∇f, Δf)
end

function contour_radial_poly(eta_in, eta_out, r, A, b)
    f(x)  = A*norm(x)^2 + b
    ∇f(x) = 2 * A * x
    Δf(x) = 2 * length(x) * A
    return contour(eta_in, eta_out, 2*A*r^2 + b, f, ∇f, Δf)
end

"""
    f(x) = ⟨a, x⟩ + b
"""
function contour_linear(etaL, etaR, x0, a, b = 0.0)
    f(x)  = dot(a, x) + b
    ∇f(x) = a
    Δf(x) = 0.0

    return contour(etaL, etaR, f(x0), f, ∇f, Δf)
end

"""
    f(x) = ∏ cos(π * xᵢ)
"""
function contour_cos(etaL, etaR, H)
    f(x)  = prod(cospi, x)
    function ∇f(x)
        if length(x) == 1
            return - π * typeof(x)(sinpi(x))
        else
            return - π * typeof(x)([prod(cospi, x[setdiff(1:end, i)]) * sinpi(x[i]) for i = eachindex(x)])
        end
    end
    Δf(x) = -length(x) * π^2 * prod(cospi, x)

    return contour(etaL, etaR, H, f, ∇f, Δf)
end