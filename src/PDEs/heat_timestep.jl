struct LinearizedImplicitEuler end
struct LinearizedTrapezoidalRule end
struct LinearizedSDIRK2
    α::Float64
    version::Int
end
LinearizedSDIRK2(α; version = 1) = LinearizedSDIRK2(α, version)
struct LinearizedExplicitEuler end
struct ImplicitEuler{NLS}
    tol::Float64
    niter::Int
end

function perform_timestep!(
    pc, t, dt,
    model::HeatEquation,
    bcon,
    mcache::HeatEquationCache,
    timestep_method::ImplicitEuler{:newton},
    method,
    linear_solver_method;
    kwargs...
)
    (; A, r, M, I, B, rm, rb, dT) = mcache
    (; c, T0, T) = pc.p
    (; niter, tol) = timestep_method

    M_corr = zero(pc.sparsity)

    # I don't exactly know what I was doing here. Probably clap * (eta - eta_i) is some sort of approximation for the gradient for something...
    for iter in 1:niter
        fill!(M_corr, 0)
        fill!(A, 0)
        fill!(r, 0)

        setvars!(pc.p, t+dt, model)

        for i = eachindex(pc)
            iszero(I[i,i]) && continue
            I[i,i] = c[i]
        end

        assemble_diffusion_system!(
            M, rm, B, rb,
            x -> model.λ(x, t + dt),
            x -> model.q(x, t + dt),
            boundary_at_timelevel(bcon, t + dt),
            method,
            pc;
            kwargs...
        )

        for i = eachindex(pc)
            isboundary(pc.p[i]) && continue
            (; c_laplace, neighbors) = pc.p[i]
            eta = pc.p.eta[neighbors]
            for (k, j) = enumerate(neighbors)
                M_corr[i, j] = c_laplace[k] * (eta[k] - eta[1])
            end
        end

        A .= I .+ B .- dt * (M .+ M_corr)
        r .= (I .+ B .- dt * M) * T .- rb .- dt * rm

        solve_linear_system!(dT, A, r, linear_solver_method)
        T  .= T - dT

        ndT = norm(dT)

        @show iter, ndT

        ndT <= tol && break

    end

    dT .= T .- T0

end

function perform_timestep!(
    pc, t, dt,
    model::HeatEquation,
    bcon,
    mcache::HeatEquationCache,
    timestep_method::ImplicitEuler{:fpi},
    method,
    linear_solver_method;
    kwargs...
)


    (; A, r, M, I, B, rm, rb, dT) = mcache
    (; c, T0, T) = pc.p
    (; niter, tol) = timestep_method

    setvars!(pc.p, t+dt, model)
    assemble_diffusion_system!(
        M, rm, B, rb,
        x -> model.λ(x, t + dt),
        x -> model.q(x, t + dt),
        boundary_at_timelevel(bcon, t + dt),
        method,
        pc;
        kwargs...
    )

    # smooth!(c, pc, method.smoothing)
    for i = eachindex(pc)
        iszero(I[i,i]) && continue
        I[i,i] = c[i]
    end

    @. A = I + B - dt * M

    for i = eachindex(pc)
        r[i] = isboundary(pc.p[i]) ? rb[i] : c[i] * T0[i] + dt * rm[i]
    end

    Told = copy(T)

    solve_linear_system!(T, A, r, linear_solver_method)

    err    = norm(T - Told)
    relerr = norm(T - Told) / (norm(T) + tol)

    @printf("FPI error = (%.4e, %.4e)", err, relerr)

    for _ = 2:niter
        setvars!(pc.p, t+dt, model)
        assemble_diffusion_system!(
            M, rm, B, rb,
            x -> model.λ(x, t + dt),
            x -> model.q(x, t + dt),
            boundary_at_timelevel(bcon, t + dt),
            method,
            pc;
            kwargs...
        )

        # smooth!(c, pc, method.smoothing)
        # for i = eachindex(pc)
        #     iszero(I[i,i]) && continue
        #     I[i,i] = c[i]
        # end

        @. A = I + B - dt * M

        # for i = eachindex(pc)
        #     r[i] = isboundary(pc.p[i]) ? rb[i] : c[i] * T0[i] + dt * rm[i]
        # end

        Told .= T
        solve_linear_system!(T, A, r, linear_solver_method)

        err    = norm(T - Told)
        relerr = err / (norm(T) + tol)

        @printf(", (%.4e, %.4e)", err, relerr)

        if err < tol || relerr < tol
            break
        end

    end
    println()

    for i = eachindex(pc)
        dT[i] = T[i] - T0[i]
    end

end

function perform_timestep!(
    pc, t, dt,
    model::HeatEquation,
    bcon,
    mcache::HeatEquationCache,
    ::LinearizedImplicitEuler,
    method,
    linear_solver_method;
    kwargs...
)

    (; A, r, M, I, B, rm, rb, dT) = mcache
    (; c, T0) = pc.p

    setvars!(pc.p, t+dt, model)
    assemble_diffusion_system!(
        M, rm,
        B, rb,
        x -> model.λ(x, t + dt),
        x -> model.q(x, t + dt),
        boundary_at_timelevel(bcon, t + dt),
        method,
        pc;
        kwargs...
    )

    # smooth!(c, pc, method.smoothing)
    for i = eachindex(pc)
        iszero(I[i,i]) && continue
        I[i,i] = c[i]
    end

    A .=  I - dt * M + B
    r .= -B * T0 + rb + dt * (rm + M * T0)

    # implicit Euler step
    solve_linear_system!(dT, A, r, linear_solver_method)

end

function perform_timestep!(
    pc, t, dt, model::HeatEquation, bcon, mcache::HeatEquationCache, ::LinearizedExplicitEuler, method, linear_solver_method; kwargs...)

    (; A, r, M, I, B, rm, rb, dT) = mcache
    (; c, T0) = pc.p

    ### first stage
    setvars!(pc.p, t, model)

    assemble_diffusion_system!(
        M, rm,
        B, rb,
        x -> model.λ(x, t),
        x -> model.q(x, t),
        boundary_at_timelevel(bcon, t + dt),
        method,
        pc;
        kwargs...
    )

    for i = eachindex(pc)
        iszero(I[i,i]) && continue
        I[i,i] = c[i]
    end

    A .= B .+ I
    r .= dt * (M * T0 .+ rm) .+ rb .- B * T0

    solve_linear_system!(dT, A, r, linear_solver_method)

end

function perform_timestep!(
    pc, t, dt,
    model::HeatEquation,
    bcon,
    mcache::HeatEquationCache,
    sdirk2::LinearizedSDIRK2,
    method,
    linear_solver_method;
    kwargs...
)

    (; A, r, M, I, B, rm, rb, dT) = mcache
    (; c, T0) = pc.p
    (; α, version) = sdirk2

    α == 0 && error("α = 0! Use explicit Euler directly!")
    α == 1 && error("α = 1! Use implicit Euler directly!")

    # (α >= 1/4 => A-stable, α = 1 ± √2/2 => L-stable)
    # Butcher-Array
    if version == 1
        # version = 1
        # α |  α   0
        # 1 | 1-α  α
        # ------------
        #   | 1-α  α

        a11 = α
        a21 = 1 - α
        a22 = α

        b1 = 1 - α
        b2 = α

        c1 = α
        c2 = 1

    elseif version == 2
        # version = 2
        # α   |  α    0
        # 1-α | 1-2α  α
        # ---------------
        #     | 1/2  1/2

        a11 = α
        a21 = 1 - 2α
        a22 = α

        b1 = 1/2
        b2 = 1/2

        c1 = α
        c2 = 1 - α
    end

    dτ11 = a11 * dt
    dτ21 = a21 * dt
    dτ22 = a22 * dt

    dt1 = b1 * dt
    dt2 = b2 * dt

    τ1 = t + c1 * dt
    τ2 = t + c2 * dt

    dτ1 = c1 * dt
    dτ2 = c2 * dt

    ### first stage
    setvars!(pc.p, τ1, model)

    assemble_diffusion_system!(
        M, rm,
        B, rb,
        x -> model.λ(x, τ1),
        x -> model.q(x, τ1),
        boundary_at_timelevel(bcon, τ1),
        method,
        pc;
        kwargs...
    )

    for i = eachindex(pc)
        iszero(I[i,i]) && continue
        I[i,i] = c[i]
    end

    A .= B .+ I .- dτ11 * M
    r .= M * T0 .+ rm .+ (rb .- B * T0) / dτ1

    k1 = copy(T0)
    solve_linear_system!(k1, A, r, linear_solver_method)

    ### second stage
    setvars!(pc.p, τ2, model)

    assemble_diffusion_system!(
        M, rm,
        B, rb,
        x -> model.λ(x, τ2),
        x -> model.q(x, τ2),
        boundary_at_timelevel(bcon, τ2),
        method, pc;
        kwargs...
    )

    for i = eachindex(pc)
        iszero(I[i, i]) && continue
        I[i, i] = c[i]
    end

    A .= I .+ B .- dτ22 * M
    r .= M * (T0 + dτ21 * k1) .+ rm .+ (rb .- B * T0) / dτ2
    k2 = copy(T0)
    solve_linear_system!(k2, A, r, linear_solver_method)

    dT .= dt1 * k1 + dt2 * k2

end

function perform_timestep!(
    pc, t, dt,
    model::HeatEquation,
    bcon,
    mcache::HeatEquationCache,
    ::LinearizedTrapezoidalRule,
    method,
    linear_solver_method;
    kwargs...
)

    (; A, r, M, I, B, rm, rb, dT) = mcache
    (; c, T0) = pc.p

    # Butcher-Array
    # 0 |  0    0
    # 1 | 1/2  1/2
    # ---------------
    #   | 1/2  1/2

    ### first stage
    setvars!(pc.p, t, model)

    assemble_diffusion_system!(
        M, rm,
        B, rb,
        x -> model.λ(x, t),
        x -> model.q(x, t),
        boundary_at_timelevel(bcon, t),
        method,
        pc;
        kwargs...
    )


    for i = eachindex(pc)
        iszero(I[i,i]) && continue
        I[i,i] = c[i]
    end

    # not really necessary step
    A .= B .+ I
    r .= M * T0 .+ rm .+ (rb .- B * T0) / dt

    k1 = copy(T0)
    solve_linear_system!(k1, A, r, linear_solver_method)

    ### second stage
    setvars!(pc.p, t + dt, model)

    assemble_diffusion_system!(
        M, rm,
        B, rb,
        x -> model.λ(x, t + dt),
        x -> model.q(x, t + dt),
        boundary_at_timelevel(bcon, t + dt),
        method, pc;
        kwargs...
    )

    for i = eachindex(pc)
        iszero(I[i, i]) && continue
        I[i, i] = c[i]
    end

    A .= I .- dt/2 * M .+ B
    r .= M * (T0 + dt/2 * k1) .+ rm .+ (rb .- B * T0) / dt

    k2 = copy(T0)
    solve_linear_system!(k2, A, r, linear_solver_method)

    ### second stage: nonlinear case
    # k2 = copy(k1)
    # k2_old = similar(k2)
    # err = 10.0
    # it  = 0

    # while err > 1e-5 && it < 10
    #     k2_old .= k2
    #     pc.p.T .= pc.p.T0 + dt / 2 * (k1 + k2)
    #     # setvars!(pc.p, t, model)
    #     pc.p.λ .= model.λ.(pc.p, t + dt)

    #     assemble_diffusion_system!(
    #         M, rm,
    #         B, rb,
    #         x -> model.λ(x, t + dt),
    #         x -> model.q(x, t + dt),
    #         boundary_at_timelevel(bcon, t + dt),
    #         method,
    #         pc;
    #         kwargs...
    #     )

    #     for i = eachindex(pc)
    #         iszero(I[i,i]) && continue
    #         I[i,i] = pc.p.c[i]
    #     end

    #     A .= I - dt / 2 * M .+ B
    #     r .= M * (pc.p.T0 + dt / 2 * k1) .+ rm + (rb - B * pc.p.T0) / dt

    #     k2 .= A \ r

    #     err = iszero(k2_old) ? norm(k2) : norm(k2 - k2_old) / norm(k2_old)
    #     it += 1

    #     @show it, err
    # end

    dT .= dt/2 * (k1 + k2)

end
