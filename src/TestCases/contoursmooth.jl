function smooth_eta_1d(etaL, etaR, xL, xR)
    xM = (xL + xR) / 2
    A = [   1.0    xL   xL^2   xL^3   xL^4   xL^5   xL^6;
            0.0   1.0    2xL  3xL^2  4xL^3  5xL^4  6xL^5;
            0.0   0.0    2.0    6xL 12xL^2 20xL^3 30xL^4;
            1.0    xR   xR^2   xR^3   xR^4   xR^5   xR^6;
            0.0   1.0    2xR  3xR^2  4xR^3  5xR^4  6xR^5;
            0.0   0.0    2.0    6xR 12xR^2 20xR^3 30xR^4;
            0.0   0.0    2.0    6xM 12xM^2 20xM^3 30xM^4;
    ]

    r = [etaL, 0.0, 0.0, etaR, 0.0, 0.0, 0.0]

    coeffs = A \ r

    function η(x)
        res = 0.0
        if x <= xL
            res = etaL
        elseif x >= xR
            res = etaR
        else
            for i = 1:7
                res += coeffs[i] * x^(i-1)
            end
        end
        return res
    end

    function η_x(x)
        res = 0.0
        if xL < x < xR
            for i = 2:7
                res += (i-1) * coeffs[i] * x^(i-2)
            end
        end
        return res
    end

    function η_xx(x)
        res = 0.0
        if xL < x < xR
            for i = 3:7
                res += (i-1) * (i-2) * coeffs[i] * x^(i-3)
            end
        end
        return res
    end

    return η, η_x, η_xx
end

"""
    f(x) = exp(-α‖x‖^2)
"""
function contour_radial_exp_smooth(eta_out, eta_in, r, h; α = 1.0)
    eta1d, eta1d_x, eta1d_xx = smooth_eta_1d(eta_in, eta_out, r - h, r + h)
    dum, u, ∇u, Δf = contour_radial_exp(eta_out, eta_in, r; α)
    η(x) = eta1d(norm(x))
    return η, u, ∇u, Δf
end

function contour_radial_poly_smooth(eta_out, eta_in, r, h, A, b)
    eta1d, eta1d_x, eta1d_xx = smooth_eta_1d(eta_in, eta_out, r - h, r + h)
    dum, u, ∇u, Δf = contour_radial_poly(eta_out, eta_in, r, A, b)
    η(x) = eta1d(norm(x))
    return η, u, ∇u, Δf
end

function contour_linear_smooth(etaL, etaR, x0, h, a, b = 0.0)
    tmp = dot(a,x0) + b

    eta1d, eta1d_x, eta1d_xx = smooth_eta_1d(etaL, etaR, tmp - h, tmp + h)
    dum, u, ∇u, Δf = contour_linear(etaL, etaR, x0, a, b)

    η(x) = eta1d(tmp + dot(a, x - x0))

    return η, u, ∇u, Δf
end