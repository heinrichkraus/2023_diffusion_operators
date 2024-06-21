include("n_strip.jl")
include("contour.jl")
include("contoursmooth.jl")
include("stefan.jl")

struct DiffusionTest{case}
    data
end

struct DiffusionTest2{case}
    u
    η
    f
    ∇u
    ∇η
    data
end

testcase(::DiffusionTest2{case}) where case = case
solution(dt::DiffusionTest2) = dt.u
diffusivity(dt::DiffusionTest2) = dt.η
source(dt::DiffusionTest2) = dt.f

function TwoStripTest(ηL, ηR, uL, uR; x0 = 0.0, xmin = -1, xmax = 1, dims = 1)
    coeffs = n_strip_test((ηL, ηR), (xmin, x0, xmax), uL, uR)
    u(p) = p.x[dims] < x0 ? coeffs[1] * p.x[dims] + coeffs[3] : coeffs[2] * p.x[dims] * coeffs[4]
    η(p) = p.x[dims] < x0 ? ηL : ηR
    f(p) = zero(precision(p))
    ∇u(p) = p.x[dims] < x0 ? coeffs[1] : coeffs[2]
    ∇η(p) = zero(precision(p))
end

DiffusionTest{case}(; kwargs...) where case = DiffusionTest{case}(values(kwargs))

testname(::DiffusionTest{case}) where case = string(case)


ContourExp(; ηIn, ηOut, r) = DiffsuionTest{:interior_radial_exp}(ηIn = ηIn, ηOut = ηOut, r = r)

#========== two strip test ==========#

TwoStripTest(; ηL, ηR, uL, uR, neumann_conservative) = DiffusionTest{:two_strip}(ηL = ηL, ηR = ηR, uL = uL, uR = uR, neumann_conservative = neumann_conservative)
function PoissonEquation(difftest::DiffusionTest{:two_strip})
    @unpack ηL, ηR = difftest.data
    η(p) = p.x[1] < 0 ? ηL : ηR
    return PoissonEquation(η, x -> 0.0)
end

function Solution(difftest::DiffusionTest{:two_strip})
    @unpack ηL, ηR, uL, uR = difftest.data
    coeffs = n_strip_test([ηL, ηR], [-1.0, 0.0, 1.0], uL, uR)
    u(p) = p.x[1] < 0 ? coeffs[1] * p.x[1] + coeffs[3] : coeffs[2] * p.x[1] + coeffs[4]
    return u
end

function BoundaryConditions(difftest::DiffusionTest{:two_strip})
    @unpack uL, uR = difftest.data

    return (
        left    = Dirichlet(uL),
        right   = Dirichlet(uR),
        top     = Neumann(0.0),
        bottom  = Neumann(0.0)
    )

end

function testcasedata!(dict::AbstractDict, difftest::DiffusionTest{:two_strip})
    @unpack ηL, ηR = difftest.data
    dict["δη"] = ηR / ηL
end



#======== three strip test ========#

ThreeStripTest(; ηL, ηM, ηR, uL, uR, neumann_conservative) = DiffusionTest{:three_strip}(ηL = ηL, ηM = ηM, ηR = ηR, uL = uL, uR = uR, neumann_conservative = neumann_conservative)
function PoissonEquation(difftest::DiffusionTest{:three_strip})
    @unpack ηL, ηM, ηR = difftest.data
    function η(p)
        if p.x[1] < -1/3
            return ηL
        elseif -1/3 <= p.x[1] <= 1/3
            ηM
        else
            ηR
        end
    end
    return PoissonEquation(η, x -> 0.0)
end

function Solution(difftest::DiffusionTest{:three_strip})
    @unpack ηL, ηM, ηR, uL, uR = difftest.data
    coeffs = n_strip_test([ηL, ηM, ηR], [-1.0, -1/3, 1/3, 1.0], uL, uR)
    function u(p)
        if p.x[1] < -1/3
            coeffs[1] * p.x[1] + coeffs[4]
        elseif -1/3 <= p.x[1] <= 1/3
            coeffs[2] * p.x[1] + coeffs[5]
        else
            coeffs[3] * p.x[1] + coeffs[6]
        end
    end
    return u
end

function BoundaryConditions(difftest::DiffusionTest{:three_strip})
    @unpack uL, uR, neumann_conservative = difftest.data

    if neumann_conservative
        return (
            left    = Dirichlet(uL),
            right   = Dirichlet(uR),
            top     = ConservativeNeumann(0.0),
            bottom  = ConservativeNeumann(0.0)
        )
    else
        return (
            left    = Dirichlet(uL),
            right   = Dirichlet(uR),
            top     = Neumann(0.0),
            bottom  = Neumann(0.0)
        )
    end

end

function testcasedata!(dict::AbstractDict, difftest::DiffusionTest{:three_strip})
    @unpack ηL, ηM, ηR = difftest.data
    dict["δη"] = ηM / ηL
    dict["δη2"] = ηR / ηM
end


#======= curved interface davydov =======#

CurvedInterface(; ηL, ηR) = DiffusionTest{:curved_interface}(ηL = ηL, ηR = ηR)
function PoissonEquation(difftest::DiffusionTest{:curved_interface})
    @unpack ηL, ηR = difftest.data
    η(p) = p.x[2] >= 2p.x[1]^3 ? ηL : ηR
    f(p) = -(120p.x[1]^4 - 24p.x[1] * (p.x[2] - 15) + 2)
    return PoissonEquation(η, f)
end

function Solution(difftest::DiffusionTest{:curved_interface})
    @unpack ηL, ηR = difftest.data
    function u(p)
        tmp = p.x[2] - 2p.x[1]^3
        ucont = tmp^2 - 30tmp
        if tmp >= 0
            return ucont / ηL
        else
            return ucont / ηR
        end
    end
    return u
end

function BoundaryConditions(difftest::DiffusionTest{:curved_interface})
    u = Solution(difftest)

    return (
        left    = Dirichlet(u),
        right   = Dirichlet(u),
        top     = Dirichlet(u),
        bottom  = Dirichlet(u)
    )

end

function testcasedata!(dict::AbstractDict, difftest::DiffusionTest{:curved_interface})
    @unpack ηL, ηR = difftest.data
    dict["δη"] = ηR / ηL
end


#====== contour cos ======#

ContourCos(; ηIn, ηOut, H) = DiffusionTest{:interior_cos}(ηIn = ηIn, ηOut = ηOut, H = H)
function PoissonEquation(difftest::DiffusionTest{:interior_cos})
    @unpack ηIn, ηOut, H = difftest.data
    η(p) = prod(cospi, p.x / 2) >= H ? ηIn : ηOut
    f(p) = dimension(p) * pi^2 / 4 * prod(cospi, p.x / 2)
    return PoissonEquation(η, f)
end

function Solution(difftest::DiffusionTest{:interior_cos})
    @unpack ηIn, ηOut, H = difftest.data
    function u(p)
        ucont = prod(cospi, p.x / 2)
        eta = if ucont >= H
            ηIn
        else
            ηOut
        end
        return (ucont - H) / eta + H
    end
    return u
end

function BoundaryConditions(difftest::DiffusionTest{:interior_cos})
    u = Solution(difftest)
    return (
        left    = Dirichlet(u),
        right   = Dirichlet(u),
        top     = Dirichlet(u),
        bottom  = Dirichlet(u)
    )
end

function testcasedata!(dict::AbstractDict, difftest::DiffusionTest{:interior_cos})
    @unpack ηIn, ηOut = difftest.data
    dict["δη"] = ηIn / ηOut
end