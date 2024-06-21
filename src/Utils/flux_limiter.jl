"""
Flux limiters \n
    ϕ_hq(r) = 2*(r + abs(r)) / (r + 3)              hquick
    ϕ_sb(r) = max(0, min(2r,1), min(r,2))           superbee
    ϕ_mm(r) = max(0, min(1,r))                      minmod
    ϕ_vl(r) = (r + abs(r)) / (1 + abs(r))           van leer
    ϕ_sw(r,β = 0.5) = max(0, min(β*r,1), min(r,2))  sweby
"""
function flux_limiter(r, method::String; β = 0.5)
    if method == "hquick" || method == "hq"
        return 2*(r + abs(r)) / (r + 3)
    elseif method == "superbee" || method == "sb"
        return max(0, min(2r,1), min(r,2))
    elseif method == "minmod" || method == "mm"
        return max(0, min(1,r))
    elseif method == "vanleer" || method == "van leer" || method == "vl"
        return (r + abs(r)) / (1 + abs(r))
    elseif method == "sweby" || method == "sw"
        return max(0, min(β*r,1), min(r,2))
    else
        error("Unknown flux limiter $method")
    end
end