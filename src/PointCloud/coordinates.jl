function coordnames(::Val{dim}) where dim
    if dim == 1
        tuple(:x)
    elseif dim == 2
        (:x, :y)
    elseif dim == 3
        (:x, :y, :z)
    else
        ntuple(i -> Symbol("x$i"), dim)
    end
end

function normalnames(::Val{dim}) where dim
    if dim == 1
        tuple(:n)
    elseif dim == 2
        (:nx, :ny)
    elseif dim == 3
        (:nx, :ny, :nz)
    else
        ntuple(i -> Symbol("n$i"), dim)
    end
end

identorzero(i::Integer) = i
identorzero(::Any) = 1

function findfirstorzero(f, values)
    iout = 0
    for (i, val) = enumerate(values)
        if f(val)
            iout = i
            break
        end
    end
    return iout
end

# coordposition(f::Symbol,  ::Val{dim}) where dim = identorzero(findfirst(isequal(f), coordnames(Val(dim))))
# normalposition(f::Symbol, ::Val{dim}) where dim = identorzero(findfirst(isequal(f), normalnames(Val(dim))))

coordposition(f::Symbol,  ::Val{dim}) where dim = findfirstorzero(isequal(f), coordnames(Val(dim)))
normalposition(f::Symbol, ::Val{dim}) where dim = findfirstorzero(isequal(f), normalnames(Val(dim)))