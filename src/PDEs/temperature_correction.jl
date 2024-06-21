abstract type TemperatureCorrection end

struct TemperatureCorrection_Off <: TemperatureCorrection end

struct TemperatureCorrection_SteepestDescent <: TemperatureCorrection
    iterations::Int
end

struct TemperatureCorrection_TH{U, Τ} <: TemperatureCorrection
    u::U
    T::Τ
end

struct TemperatureCorrection_HT{U} <: TemperatureCorrection
    u::U
    iterations::Int
end

function correct_dT!(
        ::Any,
        ::TemperatureCorrection_Off,
        ::TimeStepData,
        ::HeatEquation,
        ::PointCloud
    )
    return
end

function correct_dT!(
    dT,
    correction_method::TemperatureCorrection_HT,
    tsd::TimeStepData,
    ::HeatEquation,
    pc::PointCloud
)

    # correct dT that are too small to be considered
    dT_min, dT_max = extrema(dT)
    span_dT = dT_max - dT_min
    if span_dT > 0
        for i = eachindex(pc)
            if abs(dT[i] / span_dT) < 1e-4
                dT[i] = 0
            end
        end
    end

    # assign a weight to points that need to be corrected
    u_next = @. correction_method.u(pc.p.T + dT)
    u_curr = @. correction_method.u(pc.p.T)
    u_proj = @. u_curr + pc.p.c * dT

    w = zeros(length(pc))

    wtol = 1e-2

    for i = eachindex(pc)
        pc.p.bnd_flag[i] == :dirichlet && continue
        w[i] = pc.p.dV[i] * (abs(u_proj[i] - u_next[i]) > wtol ? 1 : 0)
    end

    # choose similar correction to steepest descent correction
    dU_predict = predict_dU(pc, tsd.Δt)

    epsilon      = sum(@. w * (u_next - u_curr)) - dU_predict
    grad_epsilon = similar(u_next)
    for i = eachindex(pc)
        grad_epsilon[i] = if dT[i] ≈ 0
            0
        else
            w[i] * (u_next[i] - u_curr[i]) / dT[i]
        end
    end

    iter = 0
    nb_maxiter = correction_method.iterations

    while norm(grad_epsilon, Inf) > 1e-2 && abs(epsilon) > 1e-12 && iter < nb_maxiter
        alpha = -epsilon / dot(grad_epsilon, grad_epsilon)
        dT[:] = dT + alpha * grad_epsilon

        iter += 1

        if iter < nb_maxiter
            # update u_next and corresponding data
            for i = eachindex(pc)
                pc.p.bnd_flag[i] == :dirichlet && continue
                w[i] = abs(u_proj[i] - u_next[i]) > wtol ? 1 : 0
            end
            u_next[:] = @. correction_method.u(pc.p.T + dT)
            epsilon   = sum(@. w * (u_next - u_curr)) - dU_predict
            for i = eachindex(pc)
                grad_epsilon[i] = if dT[i] ≈ 0
                    0
                else
                    w[i] * (u_next[i] - u_curr[i]) / dT[i]
                end
            end
        end

    end

end


function correct_dT!(
        dT,
        correction_method::TemperatureCorrection_TH,
        ::TimeStepData,
        ::HeatEquation,
        pc::PointCloud
    )

    # correct dT that are too small to be considered
    # dT_min, dT_max = extrema(dT)
    # span_dT = dT_max - dT_min
    # if span_dT > 0
    #     for i = eachindex(pc)
    #         if abs(dT[i] / span_dT) < 1e-2
    #             dT[i] = 0
    #         end
    #     end
    # end

    # dT_old = copy(dT)

    for i = eachindex(pc)
        pc.p.bnd_flag[i] == :dirichlet && continue

        T_old = pc.p.T0[i]
        # current energy level
        u = correction_method.u(T_old)

        # energy change, assuming that c is constant
        Δu = pc.p.c0[i] * dT[i]

        # find temperatures that match the expected energy change
        dT_propose = correction_method.T(u + Δu) - T_old
        dT_next    = argmin(abs, (dT_propose, dT[i]))

        if dT_next * dT[i] < 0
            mag = round(Int, log(10, abs(dT_next - dT[i])))
            @warn "Correcting in the opposite direction at point $i. Magnitude = 10^$mag."
        end

        # correction
        dT[i] = dT_next
    end

    # @show norm(dT - dT_old)

end

function correct_dT!(
        DT,
        correction_method::TemperatureCorrection_SteepestDescent,
        tsd::TimeStepData,
        model::HeatEquation,
        pc::PointCloud
    )
    correction_method.iterations == 0 && return

    dE_expected = predict_dU(pc, tsd.Δt)

    C(T) = model.c(T) # this requires c to be defined as a function of temperature also!


    # temporary stuff
    T0 = copy(pc.p.T)
    C_init = C.(T0)
    C_new  = C.(T0 + DT)

    # assuming that T_next = T0 + DT, we obtain the following energy change (approximately!)
    DE_num_pw  = @. (C_new + C_init) / 2 * DT       # trapezoidal rule, pointwise energy change
    DE_num_vol = sum(@. DE_num_pw * pc.p.dV)    # total energy change

    # apply correction
    for k = 1:correction_method.iterations
        if abs(DE_num_vol - dE_expected) > 0
            grad_DE_num_vol = [pc.p.dV[i] * C_new[i] for i = eachindex(pc)]
            alpha = - (DE_num_vol - dE_expected) / dot(grad_DE_num_vol, grad_DE_num_vol)
            DT[:] = DT + alpha * grad_DE_num_vol
            if k < correction_method.iterations
                C_new = C.(T0 + DT)
                DE_num_pw  = @. (C_new + C_init) / 2 * DT
                DE_num_vol = sum(@. DE_num_pw * pc.p.dV)
            end
        end
    end
end

function predict_dU(pc::PointCloud, Δt::Number)
    energy_sum = 0.0
    pc.p.eta .= pc.p.λ
    for i = eachindex(pc)
        if isboundary(pc.p[i])
            p = pc.p[i]
            # energy_sum += si * dot(neumann_weak(i, pc), pc.p.T[pc.p.neighbors[i]])
            c_normal = directional_derivative_row(p.n, p.x, p.h, p.neighbors,  1, pc, OneDimensionalCorrection_Default())
            energy_sum += p.dS * p.λ * dot(c_normal, pc.p.T[p.neighbors])
        else
            energy_sum += pc.p.dV[i] * pc.p.q[i]
        end
        isnan(energy_sum) && error("i = $i")
    end

    return Δt * energy_sum
end