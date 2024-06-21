
struct StefanProblem
    T_range::NTuple{2, Float64}         #T_s, T_melt
    k_range::NTuple{2, Float64}         #k_s, k_l
    s_range::NTuple{2, Float64}         #s_s, s_l
    T_phase_change::Float64             #phase change temperature
    latent_heat::Float64                #latent_heat
end


function ST_interp(x, ymin, ymax, x0, dx)
    dx == 0 && error("dx cannot be zero!")
    return (ymax + ymin) / 2 + (ymax - ymin) / 2 * tanh(3 * (x - x0) / dx)
end

function ST_dinterp(x, ymin, ymax, x0, dx)
    dx == 0 && error("dx cannot be zero!")
    3 / 2 * (ymax - ymin) / dx * sech(3 * (x - x0) / dx)^2
end

function ST_smoothed_rectfun(x, ymin, ymax, xl, xr, dx)
    ST_interp(x, ymin, ymax, xl, dx) - ST_interp(x, ymin, ymax, xr, dx)
end

function approximate_interface(p, T_m)
    x_values = map(x -> x[1], p.x)
    T_values = p.T

    x_left, x_right = extrema(x_values)

    for (x, T) = zip(x_values, T_values)
        x_right = (T <= T_m && x < x_right) ? x : x_right
        x_left  = (T >= T_m && x > x_left)  ? x : x_left
    end

    return (x_left + x_right) / 2
end

function phase_change_region(p)
    pc_points = findall(!iszero, p.within_pc)
    x_values = map(x -> x[1], p.x[pc_points])
    return isempty(x_values) ? (0.0, 0.0) : extrema(x_values)
end

function pointcloud_stefan(
    prob::StefanProblem,
    tend;                               # final time
    δ_rand             = 0.0,           # randomization parameter
    extension_factor   = 2.0,           # extension factor in x-direction
    aspect_ratio_scale = 32.0,          # aspect ratio scaler
    aspect_ratio       = (6, 1),        # aspect ratio
    time_delay         = 0.0,
)

    Random.seed!(122023712023)      # reproducibility

    # unpack
    T_s, T_l    = prob.T_range
    k_s, k_l    = prob.k_range
    s_s, s_l    = prob.s_range
    T_m         = prob.T_phase_change
    latent_heat = prob.latent_heat

    nb_points = aspect_ratio_scale .* aspect_ratio

    ## Analytic reference solution
    _, X = stefan_solution(
        (T_s, T_l), (k_s, k_l), (s_s, s_l), T_m, latent_heat
    )

    X_offset = X(time_delay)

    ## Point cloud generation
    xmin = 0.0
    ymin = 0.0
    xmax = round((extension_factor * X(tend) - X_offset)/aspect_ratio[2], sigdigits=1)
    ymax = round(xmax / aspect_ratio[1], sigdigits = 1)

    δx = (xmax - xmin) / nb_points[1]
    δy = (ymax - ymin) / nb_points[2]
    Δx = min(δx, δy)

    if ymax < 5Δx || xmax < 10Δx
        error("Something went wrong with the resolution")
    end

    pc = rectangle(xmin, xmax, ymin, ymax, Δx, δ_rand)

    return pc

end

stefan_time_error(error)            = sum(error)/length(error)
stefan_interface_error(X, X_approx) = norm(X - X_approx, Inf)/norm(X, Inf)

function mindist(pc)
    Δx = Inf
    for i = eachindex(pc), j = pc.p.neighbors[i]
        j == i && continue
        Δx = min(Δx, norm(pc.p.x[j] - pc.p.x[i]))
    end
    return Δx
end

"""
energy_correction = {:th_curve, :none, :steepest_descent}
"""
function solve_stefan!(
    pc::PointCloud,
    prob::StefanProblem;
    T_ϵ             = 0.0,
    k_smoothing     = 0,                                # number of smoothing cycles
    timespan        = (0.0, 100.0),
    save_interval   = (:constant, 1),                   # :constant / :time
    timestep_method = ImplicitEuler{:fpi}(1e-4, 50),    # LinearizedImplicitEuler also possible
    timestep_size   = (:cfl, 10.0),                     # (:cfl, num), (:constant, num)...
    nb_maxiter      = typemax(Int),
    s_model         = :disc,
    k_model         = :linear,
    time_delay      = first(timespan),                  # delay of phase change start
    energy_correction     = :th_curve,
    correction_refinement = 100,
    kwargs...
)

    # unpack
    T_s, T_l    = prob.T_range
    k_s, k_l    = prob.k_range
    s_s, s_l    = prob.s_range
    T_m         = prob.T_phase_change
    latent_heat = prob.latent_heat

    if !iszero(latent_heat) && iszero(T_ϵ)
        error("Need phase change region for nonvanishing latent heat!")
    end

    # time discretization
    tstart, _ = timespan              # t limits (also for xmax, ymax calculation...)

    ## Analytic reference solution
    sol_stefan, X = stefan_solution(
        (T_s, T_l), (k_s, k_l), (s_s, s_l), T_m, latent_heat
    )

    T_ref(x::Number, t) = sol_stefan(x, t)
    T_ref(p, t)         = sol_stefan(p.x[1], t)

    # (possibly) delay phase change stuff
    X_offset = X(time_delay)

    ## Model and point cloud
    c(T) = if s_model == :disc || iszero(T_ϵ)
        T < T_m ? s_s : s_l
    elseif s_model == :cont || s_model == :cont2
        ST_interp(T, s_s, s_l, T_m, T_ϵ)
    end

    s(T) = if iszero(latent_heat)
        c(T)
    elseif s_model == :disc
        c(T) + (abs(T - T_m) < T_ϵ) * latent_heat / 2T_ϵ
    elseif s_model == :cont
        c(T) + latent_heat * ST_dinterp(T, 0, 1, T_m, 2T_ϵ)
    elseif s_model == :cont2
        c(T) + latent_heat * ST_smoothed_rectfun(T, 0, 1, T_m-T_ϵ, T_m+T_ϵ, T_ϵ/4) / 2T_ϵ
    else
        error("Unknown model!")
    end

    k(T) = if T_ϵ == 0 || k_model == :disc
        T < T_m ? k_s : k_l
    elseif k_model == :cont
        ST_interp(T, k_s, k_l, T_m, T_ϵ)
    elseif k_model == :linear
        if abs(T-T_m) < T_ϵ
            (T-T_m) * (k_l - k_s) / 2T_ϵ + (k_l + k_s) / 2
        elseif T > T_m
            k_l
        else
            k_s
        end
    end

    s(p, t) = s(p.T)
    k(p, t) = k(p.T)
    q(p, t) = 0
    T(p, t) = T_ref(p.x[1] + X_offset, t)

    bcon = (
        left   = Dirichlet(T),
        right  = Dirichlet(T),
        top    = ConservativeNeumann(0),
        bottom = ConservativeNeumann(0),
    )

    model = HeatEquation(s, k, q, p -> T(p, tstart))

    ## timestep size
    prepare_timestep!(pc)

    Δx = mindist(pc)

    Δt = if first(timestep_size) == :cfl
        last(timestep_size) * Δx^2
    elseif first(timestep_size) == :constant
        last(timestep_size)
    end

    save_int = if save_interval[1] == :constant
        save_interval[2]
    elseif save_interval[1] == :time
        max(1, round(Int, save_interval[2]/Δt))
    else
        NaN
    end

    ## Correction algorithm setup
    correction_algorithm = if energy_correction == :none || iszero(latent_heat)
        TemperatureCorrection_Off()
    elseif energy_correction == :th_curve
        T_range =
        if s_model == :cont || s_model == :cont2
            range(T_s, T_l, length = 10*correction_refinement+1)
        elseif s_model == :disc
            union(
                range(T_s, T_m - T_ϵ, length=2),
                range(T_m - T_ϵ, T_m + T_ϵ, length=3),
                range(T_m + T_ϵ, T_l, length=2)
            )
        end

        E_of_T = antiderivative(s, T_range)
        T_of_E = inverse(E_of_T)
        u(T) = E_of_T(T)
        TemperatureCorrection_TH(E_of_T, T_of_E)
    elseif energy_correction == :steepest_descent
        TemperatureCorrection_SteepestDescent(10)
    else
    end

    ## Solve
    integrations = solve!(
        pc, model, bcon;
        method=DiffusionSystem(
            Smoothing(x -> exp(-5x), k_smoothing, exclude = (:left, :right)),
            DiffusionOperator_Single(DivEtaGrad_ScaledLaplacian(:harmonic_mean, Laplace_WLSQ(order = 2)))
        ),
        correction_algorithm,
        timespan = timespan,
        Δt_max=Δt,
        save_items = (
            T_ref        = (p, t) -> T.(p, t),
            T_resid      = (p, t) -> p.T_ref - p.T,
            T_resid_abs  = (p, t) -> abs.(p.T_resid),
            within_pc    = (p, t) -> map(T -> abs(T - T_m) <= T_ϵ, p.T),
        ),
        save_interval=save_int, integrations = (
            error_li  = (p, t) -> norm(p.T_resid, Inf),
            error_l2  = (p, t) -> norm(p.T_resid, pc, 2) / sqrt(sum(p.dV)),
            X         = (p, t) -> max(X(t)-X_offset, 0.0),
            X_approx  = (p, t) -> approximate_interface(p, T_m),
            X_min     = (p, t) -> phase_change_region(p)[1],
            X_max     = (p, t) -> phase_change_region(p)[2],
        ),
        timestep_method,
        nb_maxiter,
        kwargs...
    )

    stats = (; dt=Δt)

    return stats, integrations

end

function stefan_solution(prob::StefanProblem)
    return stefan_solution(
        prob.T_range,
        prob.k_range,
        prob.s_range,
        prob.T_phase_change,
        prob.latent_heat
    )
end

function stefan_solution(
    T_range,            #temperature range
    λ_range,            #jumping heat conductivity
    s_range,            #jumping volumetric heat conductivity
    T_phase_change,     #phase change teperature
    latent_heat         #latent heat
)
    T_cold, T_hot = T_range
    λ_cold, λ_hot = λ_range
    s_cold, s_hot = s_range

    α_hot  = λ_hot / s_hot
    α_cold = λ_cold / s_cold
    α_frac = α_hot / α_cold

    # the problem with and without latent heat is basically the same, differing by this factor
    lh_factor = !iszero(latent_heat) ? 1.0 / latent_heat : 1.0

    # Stefan numbers
    ST_hot  = s_hot * (T_hot - T_phase_change) * lh_factor
    ST_cold = s_cold * (T_phase_change - T_cold) * lh_factor

    nu = sqrt(α_frac)

    f(x) = ST_hot * nu * erfc(nu * x) * exp(nu^2 * x^2) -
        ST_cold * erf(x) * exp(x^2) -
        nu * erfc(nu * x) * erf(x) * sqrt(pi) * x *
        exp(nu^2 * x^2) * exp(x^2) * (!iszero(latent_heat))

    # nonlinear problem in different formulation to test if solution is good enough
    f2(x) = ST_hot / (exp(x^2) * erf(x)) - ST_cold / (nu * exp(nu^2 * x^2) * erfc(nu * x)) -
            x * sqrt(pi) * (!iszero(latent_heat))

    # the guess (0.7) should ideally depend on the model parameters
    x_left  = 0.0
    x_right = 0.0
    x_gain  = 0.1
    while f(x_left) * f(x_right) >= 0
        x_left   = x_right
        while isinf(f(x_right+x_gain)) || isnan(f(x_right+x_gain))
            x_gain *= 0.5
        end
        x_right += x_gain
    end
    λ = bisection(f, x_left, x_right)

    @show λ, f(λ), f2(λ)

    conds = [λ <= 0, abs(f(λ)) > 1e-10, abs(f2(λ)) > 1e-6]

    if any(conds)
        conds_violated = findall(conds)
        error("Conditions $conds_violated violated! NO FURTHER COMPUTATION!")
    end

    X(t) = 2λ * sqrt(α_hot * t)

    T_L(x, t) = T_hot - (T_hot - T_phase_change) * erf(x / 2sqrt(α_hot * t)) / erf(λ)
    T_R(x, t) = T_cold + (T_phase_change - T_cold) * erfc(x / 2sqrt(α_cold * t)) / erfc(λ * sqrt(α_frac))

    function T(x::Number, t)
        if iszero(t)
            return x <= X(t) ? T_hot : T_cold
        else
            return x <= X(t) ? T_L(x, t) : T_R(x, t)
        end
    end

    return (
        solution = T,
        interface = X
    )
end
