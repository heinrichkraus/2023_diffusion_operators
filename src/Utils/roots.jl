""" Find roots with Newton method """
function newton(x0, f::Function, df::Function; tol::Real=1e-10, maxiter::Int=1000, verbose=false)
    n_var = length(x0)

    f0  = f(x0)
    df0 = df(x0)

    if n_var != length(f0)
        error("Size of x0 and f(x0) does not match.")
    elseif n_var != size(df0,1) || n_var != size(df0,2)
        error("Size of x0 and df(x0) does not match.")
    end

    Δx  = zeros(Float64, n_var)
    err = norm(f0)
    cnt = 0

    while err > tol && cnt < maxiter
        #Newton iteration
        Δx = - df0 \ f0
        x0 = x0 + Δx

        #Update values
        f0  = f(x0)
        df0 = df(x0)

        #Error calculation
        err = norm(f0)

        cnt += 1
    end

    verbose && println("Exiting newton iteration after $cnt iterations with an error of $err")

    return x0
end

""" Find root with bisection method """
function bisection(f, a, b; tol=1e-10, maxiter=1000)
    a >= b && error("a >= b!")

    fa = f(a)
    fb = f(b)

    any(isnan, (fa, fb)) && error("NaN found!")
    any(isinf, (fa, fb)) && error("Inf found!")

    fa * fb > 0 && error("f(a) * f(b) > 0, fa = $fa, fb = $fb")
    fa == 0 && return a
    fb == 0 && return b

    x = (a + b) / 2
    fx = f(x)
    iter = 0
    while abs(fx) > tol && iter < maxiter
        abs(fa) < tol && return a
        abs(fb) < tol && return b

        a, b = fx * fa > 0 ? (x, b) : (a, x)

        x = (a + b) / 2
        fx = f(x)
        fa = f(a)
        fb = f(b)

        iter += 1

        any(isnan, (fa, fb, fx)) && error("NaN found in iteration $iter.")
        any(isinf, (fa, fb, fx)) && error("Inf found in iteration $iter.")
    end

    return x
end