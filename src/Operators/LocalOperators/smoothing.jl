struct Smoothing{F,TUP}
    f::F
    n::Integer
    exclude::TUP
end

Smoothing(f, n; exclude = ()) = Smoothing(f, n, exclude)
Smoothing() = Smoothing(one, 0)

"""
Smoothes vector v n times with weight function w and radius δ.\n
Weights are calculated as w_ij = w(‖x_i - x_j‖ / δ * h) with δ > 0.
"""
function smooth!(v, pc, w, n, exclude)
    n <= 0 && return

    W = zero(pc.sparsity)

    val         = 0.0
    sum_weight  = 0.0

    for i = eachindex(pc)
        sum_weight = 0.0

        if pc.p.label[i] ∈ exclude
            W[i,i] = 1.0
        else
            for j = pc.p.neighbors[i]
                val         = w(norm(pc.p.x[i] - pc.p.x[j]) / pc.p.h[i])
                W[i,j]      = val
                sum_weight += val
            end

            scale_matrix_row!(W, 1 / sum_weight, i, pc.p.neighbors[i])
        end
    end

    for _ = 1:n
        v[:] = W * v
    end

end

smooth!(v, pc, so::Smoothing) = smooth!(v, pc, so.f, so.n, so.exclude)