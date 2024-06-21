using LinearAlgebra
function gradient_descent!(x, f, ∇f; max_iter = 100, abstol = 1e-6)
    x_last  = copy(x)
    f_curr = f(x)
    gf_curr = ∇f(x)
    gf_last = copy(gf_curr)

    w = 1

    iter = 0
    while norm(gf_curr) > abstol && iter < max_iter
        iter += 1

        f_curr = f(x)
        @show iter, w, norm(gf_curr), f_curr

        #cache old values
        copyto!(x_last, x)
        gf_last[:] = gf_curr

        #update
        copyto!(x, x_last - w * gf_last)

        #compute new values
        gf_curr = ∇f(x)

        w = abs(dot(x - x_last, gf_curr - gf_last)) / norm(gf_curr - gf_last)^2
    end

end

A =
[2.0 1.0 0.0
1.0 2.0 1.0
0.0 1.0 2.0]

b = [1.0, 2.0, 3.0]

f(x) = 1/2 * dot(A*x, x) - dot(b, x)
∇f(x) = A*x - b

x0 = zero(b)
gradient_descent!(x0, f, ∇f, abstol=1e-5)
@show x0