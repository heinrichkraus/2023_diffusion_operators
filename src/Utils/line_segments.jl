function line_segments_intersect(p1, v1, p2, v2)
    # check if two curves γi(t) = pi + t * vi, i = 1,2, intersect for t1, t2 ∈ [0,1]
    if v1 == zero(v1) || v2 == zero(v2)
        error("Directions cannot be zero!")
    end

    is_parallel = false
    n1 = v1 / norm(v1)
    n2 = v2 / norm(v2)

    if norm(n1 - n2) < 1e-12 || norm(n1 + n2) < 1e-12
        is_parallel = true
    end

    if is_parallel
        #check if p1 lies on γ2 and p2 lies on γ1
        if point_in_line(p1, p2, v2) || point_in_line(p2, p1, v1)
            return true
        else
            return false
        end
    else
        t  = [-v1 v2] \ (p1 - p2)
        if 0 <= t[1] <= 1 && 0 <= t[2] <= 1
            return true
        else
            return false
        end
    end
end

function point_in_line(p, u, v)
    # checks if p is in line u + t * v for t ∈ [0, 1]
    if v[1] != zero(v[1])
        i = 1
        j = 2
    else
        j = 1
        i = 2
    end

    # solve for t in i-th coordinate
    λ = (p[i] - u[i]) / v[i]

    if λ > 1 || λ < 0
        return false
    end

    # check λ in j-th coordinate
    if abs(u[j] + λ * v[j] - p[j]) < 1e-12
        return true
    else
        return false
    end
end