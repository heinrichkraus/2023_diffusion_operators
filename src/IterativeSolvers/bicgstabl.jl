Base.@kwdef struct LinearSolver_bicgstabl{PCT} <: LinearSolverType
    abstol::Float64 = 1e-14
    reltol::Float64 = 1e-14
    l::Int = 2
    precond::PCT = IncompleteLU.ilu
    verbose = false
end

struct DiagonalPreconditioning end

function solve_linear_system!(x, A, b, solver::LinearSolver_bicgstabl)
    (; abstol, reltol, l, precond, verbose) = solver

    if isdiag(A) || istril(A) || istriu(A)
        ldiv!(x, factorize(A), b)
        return
    end

    diag_A = diag(A)

    if any(iszero, diag_A)
        zero_idx = findall(iszero, diag_A)
        @warn "Encountered a zero on the diagonal for points $(zero_idx)!"
    end

    if precond == IterativeSolvers.Identity()
        IterativeSolvers.bicgstabl!(x, A, b, l; abstol, reltol, verbose)
    elseif precond == DiagonalPreconditioning()
        Dinv = Diagonal(1.0 ./ diag_A)
        IterativeSolvers.bicgstabl!(x, Dinv*A, Dinv*b, l; abstol, reltol, verbose)
    else
        IterativeSolvers.bicgstabl!(x, A, b, l; abstol, reltol, verbose, Pl = precond(A))
    end
end
