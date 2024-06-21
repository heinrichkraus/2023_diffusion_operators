function n_strip_test(
    η_vals,
    x_vals,
    uL, uR
)

    n_strip = length(η_vals)

    if !issorted(x_vals)
        error("sol_n_strip: x-values need to be sorted")
    end

    if length(x_vals) != n_strip + 1
        error("sol_n_strip: Number of interfaces is $(length(x_vals)) but needs to be $(n_strip + 1)")
    end

    A = zeros(2 * n_strip, 2 * n_strip)
    b = zeros(2 * n_strip)

    #reserve last rows in blocks for boundary conditions
    A[n_strip,1]           = x_vals[1]
    A[n_strip,n_strip + 1] = 1.0
    b[n_strip]             = uL

    A[2*n_strip,n_strip]   = x_vals[end]
    A[2*n_strip,2*n_strip] = 1.0
    b[2*n_strip]           = uR

    for i = 1:(n_strip-1)
        #set jump conditions
        A[i,i]           =  x_vals[i+1]
        A[i,i+1]         = -x_vals[i+1]
        A[i,i+n_strip]   =  1.0
        A[i,i+1+n_strip] = -1.0
        b[i]             =  0.0

        #set flux jump conditions
        A[i+n_strip, i]     =  η_vals[i]
        A[i+n_strip, i+1]   = -η_vals[i+1]
        b[i+n_strip]        =  0.0
    end

    coeffs = A \ b

    # function u(x,y)
    #     for i = 1:n_strip
    #         if x_vals[i] <= x <= x_vals[i+1]
    #             return coeffs[i] * x + coeffs[i + n_strip]
    #         end
    #     end
    # end

    # function η(x,y)
    #     for i = 1:n_strip
    #         if x_vals[i] <= x <= x_vals[i+1]
    #             return η_vals[i]
    #         end
    #     end
    # end

    # u_x(x,y) = 1 / η(x,y)
    # u_y(x,y) = 0.0

    # Δf(x,y) = 0.0
    # ff(x,y) = 0.0

    return coeffs
end

function five_strip_test_trask(ymin, ymax)
    η_vals = [16.0, 6.0, 1.0, 10.0, 2.0]
    y_vals = collect(range(ymin, ymax, length = 6))

    u(x,y)   = 1 - x
    u_x(x,y) = -1
    u_y(x,y) = 0

    Δf(x,y) = 0
    ff(x,y) = 1 - x

    function η(x,y)
        for i = 1:5
            if y_vals[i] <= y <= y_vals[i+1]
                return η_vals[i]
            end
        end
    end

    return η, u, u_x, u_y, Δf, ff
end