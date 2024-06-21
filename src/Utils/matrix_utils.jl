function Base.setindex!(M::SparseMatrixCSC{Tv, Ti}, vals::AbstractVector{Tv}, irow::Int, pos::Vector{Ti}) where {Tv <: AbstractFloat, Ti <: Integer}
    set_matrix_row!(M, vals, irow, pos)
end

function Base.setindex!(M::SparseMatrixCSC, vals::AbstractVector, i::Int, pc::PointCloud)
    set_matrix_row!(M, vals, i, pc.p.neighbors[i])
end

function Base.getindex(M::SparseMatrixCSC, i::Int, pc::PointCloud)
    return M[i,:].nzval
end

function set_matrix_row!(M::AbstractMatrix{Tv}, vals::AbstractVector{Tv}, irow::Int, pos::Vector{Ti}) where {Tv, Ti}
    if length(vals) == length(pos)
        for k = 1:length(pos)
            M[irow, pos[k]] = vals[k]
        end
    elseif length(vals) == size(M,2)
        for k = 1:length(pos)
            M[irow, pos[k]] = vals[pos[k]]
        end
    else
        error("")
    end
end

function scale_matrix_row!(M::AbstractMatrix{Tv}, λ, irow::Int, pos::Vector{Ti}) where {Tv, Ti}
    for k = 1:length(pos)
        M[irow, pos[k]] *= λ
    end
end

function add_to_matrix_row!(M::AbstractMatrix{Tv}, vals::AbstractVector{Tv}, irow::Int, pos::Vector{Ti}) where {Tv, Ti}
    if length(vals) == length(pos)
        for k = 1:length(pos)
            M[irow, pos[k]] += vals[k]
        end
    elseif length(vals) == size(M,2)
        for k = 1:length(pos)
            M[irow, pos[k]] += vals[pos[k]]
        end
    else
        error("")
    end
end

function check_diagonal_dominance(M::AbstractMatrix)
    # isdd = zeros(Bool, size(M, 1))

    # for i = 1:size(M,1)
    #     mii = M[i,i]
    #     row = M[i,:]

    #     rowerr = sum(abs.(row) / abs(mii)) - 2
    #     err = max(rowerr, -mii, 0)

    #     isdd[i] = err > 1e-12 ? false : true
    # end

    # return isdd
    return map((i, v) -> check_diagonal_dominance(i, v), axes(M, 1), eachrow(M))
end

function check_diagonal_dominance(i, v::AbstractVector; dd_tol = 1e-12)
    iszero(v[i]) && return iszero(v)
    return sum(abs.(v) / abs(v[i])) <= 2 + dd_tol ? true : false
end

function diagonal_dominance_error(i, v::AbstractVector)
    return v[i] == zero(v[i]) ? sum(abs, v) : sum(abs, v) / abs(v[i]) - 2
end

function diagonal_dominance_error2(i, v::AbstractVector)
    return sum(abs, v) - 2abs(v[i])
end

function diagonal_dominance_error3(i, v::AbstractVector)
    vi = v[i]
    dderr = 0.0
    for j = eachindex(v)
        j == i && continue
        dderr = max(dderr, v[j] / vi)
    end
    return dderr
end

function check_inverse_positive(M::SparseMatrixCSC)
    return check_inverse_positive(Array(M))
end

function check_inverse_positive(M::Matrix)
    n_rows = size(M,1)

    is_m_mat_row = zeros(Int, n_rows)

    invM = inv(M)
    for i = 1:n_rows
        is_m_mat_row[i] = 1
        for j = 1:n_rows
            if invM[i,j] < -1e-6
                is_m_mat_row[i] = 0
                break
            end
            if isnan(invM[i,j]) || isinf(invM[i,j])
                is_m_mat_row[i] = -1
                break
            end
        end
    end

    return is_m_mat_row
end

function nullspace_custom(K, W; tol = 1e-5, idx = 1:size(K, 2))
    C0 = zeros(size(K, 2), 0)
    r0 = zeros(size(K, 1) + 1)
    KK = [K; zeros(size(K, 2))']
    r0[end] = 1
    for k = idx
        KK[end, :] = [l == k for l = 1:size(K, 2)]
        C0 = hcat(C0, leastsquares(KK, W, r0, DD_Off()))
    end
    QR = qr(C0)
    nzcols = findall(==(true), map(>(abs(tol)), abs.(diag(QR.R))))
    1 ∉ nzcols && @warn "I removed the diagonal entry!"
    return C0[:, nzcols]
end
