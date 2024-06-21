function leastsquares(K, W::Diagonal, r::AbstractVector, dd::AbstractOneDimensionalCorrection)
    #solution that satisfies consistency conditions
    vcons = leastsquares(K, W, r, DD_Off())

    # Perform DD correction only if operator is not DD because correction could break the diagonal dominance
    if check_diagonal_dominance(1, vcons)
        return vcons
    end

    #vector for diagonal dominance
    K = [K; [j == 1 for j = 1:size(K,2)]']
    s = [zero(r)..., 1.0]
    vdiag = leastsquares(K, W, s, DD_Off())

    #compute parameter for maximal diagonal dominance
    return vcons + α_opt(vcons, vdiag, 1, dd) * vdiag
end

function leastsquares(K, W::Diagonal, r::AbstractMatrix, dd::AbstractOneDimensionalCorrection)
    nfuns = size(r, 2)
    s = fill(-4.0, 1, nfuns)

    #prepare test matrix and rhs for diagonal dominance
    K = [K; [j == 1 for j = 1:size(K,2)]']
    r = [[r; s] [zeros(size(r,1)); 1.0]]

    vsol = leastsquares(K, W, r, DD_Off())

    vcons = vsol[:,1:end-1] #solution that satisfies test function property (consistency conditions)
    vdiag = vsol[:,end] #vector for diagonal dominance

    #compute parameter for maximal diagonal dominance
    return hcat(map(v -> v + α_opt(v, vdiag, 1, dd) * vdiag, eachcol(vcons))...)
end

function leastsquares(K, W::Diagonal, r::AbstractVector, dd::AbstractMultiDimensionalCorrection)
    c = leastsquares(K, W, r, OneDimensionalCorrection_Default())
    H = nullspace(K)
    α = α_opt(c, H, 1, dd)
    return c + H * α
end

function leastsquares(K, W::Diagonal, r, ::DD_Off)
    H = W ^ 2 * transpose(K)
    T = (K * W) * (W * transpose(K))
    return H * (T \ r)
end
