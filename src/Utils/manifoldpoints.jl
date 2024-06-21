function mani_circle(pmid::SVector{2}, r, dx)
    N = round(Int, 2π / acos((2r^2 - dx^2) / 2r^2))
    Φ = range(0, 2π, length = N)
    return [r * SVector(cos(φ), sin(φ)) + pmid for φ ∈ Φ[1:end-1]]
end

function mani_circle(x0, y0, r, dx)
    return mani_circle(SVector(x0, y0), r, dx)
end

function mani_curve(γ::Function, t0, t1, dt)
    n_points = Int(floor((t1 - t0) / dt))
    p = zeros(typeof(γ(t0)), n_points)

    if t1 < t0
        return p
    end

    t   = t0
    cnt = 1
    while t < t1
        p[cnt] = Float64.(γ(t))

        cnt += 1
        if abs(t + dt - t1) < dt
            t = t1
        else
            t += dt
        end
    end

    return p
end

function mani_rectangle(xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64, dx::Float64)
    bottom_left   = SVector(xmin, ymin)
    bottom_right  = SVector(xmax, ymin)
    top_left      = SVector(xmin, ymax)
    top_right     = SVector(xmax, ymax)

    bottom  = mani_line(bottom_left, bottom_right, dx, false)
    right   = mani_line(bottom_right, top_right, dx, false)
    top     = mani_line(top_right, top_left, dx, false)
    left    = mani_line(top_left, bottom_left, dx, false)

    return [bottom_left, bottom..., bottom_right, right..., top_right, top..., top_left, left...]
end

mani_rectangle(xmin, xmax, ymin, ymax, dx) = mani_rectangle(promote(xmin, xmax, ymin, ymax, dx)...)

function mani_line(pstart, pend, dx, ep::Bool = true)
    dirvec = pend - pstart
    # dist   = norm(dirvec)
    # t      = range(0.0, stop = 1.0, step = dx / dist) |> collect
    # if t[end] != 1.0
    #     push!(t, 1.0)
    # end

    dt = dx / norm(dirvec)

    if ep == true
        # n_points = length(t)
        # i_start  = 1
        # i_end    = n_points
        t0 = 0.0
        t1 = 1.0
    else
        # n_points = length(t) - 2
        # i_start  = 2
        # i_end    = length(t) - 1
        t0 = dt
        t1 = 1.0 - dt
    end

    return [pstart + t * dirvec for t ∈ range(t0, t1, step = dt)]
    # return mani_curve(γ, t[i_start:i_end])
    # return mani_curve(γ, t0, t1, dt)
end