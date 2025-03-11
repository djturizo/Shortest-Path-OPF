module ShortestPathOPF

using LinearAlgebra
import PowerModels as PM, SparseArrays as SA
import Ipopt as Ipopt
import BlockArrays as BA, BlockBandedMatrices as BBM
# import JSON

using Infiltrator
Infiltrator.toggle_async_check(false)
# Infiltrator.clear_disabled!()

include("power_flow_rect_qc.jl")
import .PowerFlowRectQC as QC

# include("interior_point_solver.jl")
# import .InteriorPointSolver as IPS
# include("opf_wrapper.jl")
# import .OPFWrapper as OPF

# We cannot alias the 'Distributed' pkg,
# otherwise the workers will not recognize it
using Distributed
import Hwloc as hw
global ENABLE_PARALLEL = false
global USE_PARALLEL = false
const VERBOSE = true


function fullprint(var)
    if VERBOSE
        show(stdout, "text/plain", var)
        println()
    end
    return
end


function parallel_disabled_msg()
    @info "Parallel functionalities are disabled."
end

function open_parallel(n_free_cores::Int64=0)
    if ENABLE_PARALLEL
        global USE_PARALLEL = true
        if n_free_cores <= 0
            # It should be one more workers, but that gives errors
            n_free_cores = 0 + hw.num_physical_cores() - nprocs()
        end
        if n_free_cores > 0
            addprocs(n_free_cores)
        end
        @everywhere begin
            if myid() != 1
                # we are not in the main process, we need to
                # activate the environment
                import Pkg
                Pkg.activate(pwd())
                include("shortest_path.jl")
                import .ShortestPathOPF as SP
            end
        end
    else
        parallel_disabled_msg()
    end
end

function close_parallel()
    if ENABLE_PARALLEL
        rmprocs(procs())
        global USE_PARALLEL = false
    else
        parallel_disabled_msg()
    end
end

function enable_parallel()
    global ENABLE_PARALLEL = true
end

function disable_parallel()
    if USE_PARALLEL
        close_parallel()
    end
    global ENABLE_PARALLEL = false
end


struct TestCase
    case_dir::String
    ind_u::Vector{Int64}
    Q::Union{Matrix{Float64},Nothing}
    u_center::Union{Vector{Float64},Nothing}
    qc_data
    ind
    c
    opf_u
    opf_x
    opf_fobj
    case_input
    pf_solver
    get_jac
    J_c::SA.SparseMatrixCSC{Float64,Int64}
    J_k::Vector{SA.SparseMatrixCSC{Float64,Int64}}
    PdQd::Vector{Float64}
    mpc
end

struct CaseFunctions
    point_oracle
    path_oracle
    f_obj
end


function get_endpoints(mpc, case_data, tol_inner, rng)
    # # Compute start point as the solution of the minimum loss problem
    # mpc_mod = deepcopy(mpc)
    # # # Get start point from randomly modified OPF problem
    # # println(rng.seed)
    # # for (key, _) in mpc_modified["gen"]
    # #     mpc_modified["gen"][key]["cost"] .*= rand(rng, Float64)*2 - 1
    # # end
    # # Losses are minimized by minimizing the total generated power
    # for (key, _) in mpc_mod["gen"]
    #     mpc_mod["gen"][key]["cost"] = [0.0, 1.0, 0.0]
    # end
    # case_qc_mod = QC.compute_qc(mpc_mod, !true)
    # eval_f, eval_g!, eval_grad_f!, eval_jac_g!, eval_h!, nv, v_L, v_U, nc,
    #     c_L, c_U, nele_jac, nele_hess, u_0_red, x_0 = OPF.make_ipopt_callbacks(
    #     case_dir, case_qc_mod, tol_inner, tol_pf, iter_max_pf)
    # v0 = vcat(u_0_red, x_0)

    # # # Test callbacks
    # # fv = eval_f(v0)
    # # g = fill(NaN, nc)
    # # eval_g!(v0, g)
    # # grad_f = fill(NaN, nv)
    # # eval_grad_f!(v0, grad_f)
    # # rows_jac = -ones(Int32, nele_jac)
    # # cols_jac = -ones(Int32, nele_jac)
    # # values_jac = fill(NaN, nele_jac)
    # # eval_jac_g!(v0, rows_jac, cols_jac, nothing)
    # # eval_jac_g!(v0, rows_jac, cols_jac, values_jac)
    # # rows_hess = -ones(Int32, nele_hess)
    # # cols_hess = -ones(Int32, nele_hess)
    # # values_hess = fill(NaN, nele_hess)
    # # obj_factor = 1.0
    # # lambda = ones(Float64, nc)
    # # eval_h!(v0, rows_hess, cols_hess, obj_factor, lambda, nothing)
    # # eval_h!(v0, rows_hess, cols_hess, obj_factor, lambda, values_hess)
    # # @infiltrate
    # # return

    # prob = Ipopt.CreateIpoptProblem(nv, v_L, v_U, nc, c_L, c_U, nele_jac,
    #     nele_hess, eval_f, eval_g!, eval_grad_f!, eval_jac_g!, eval_h!)
    # prob.x = v0
    # # Set Ipopt options
    # Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "exact")
    # Ipopt.AddIpoptNumOption(prob, "tol", tol_inner)
    # Ipopt.AddIpoptIntOption(prob, "max_iter", iter_max_outer)
    # Ipopt.AddIpoptNumOption(prob, "constr_viol_tol", tol_inner)
    # Ipopt.AddIpoptNumOption(prob, "acceptable_tol", tol_inner)
    # Ipopt.AddIpoptNumOption(prob, "acceptable_constr_viol_tol", tol_inner)
    # solve_status = Ipopt.IpoptSolve(prob)
    # if solve_status != 0
    #     error("OPF problem could not be solved.")
    # end
    # u_start = prob.x[1:dim_u]
    # # New base case is the start point
    # u0[ind_u] = u_start
    # # Clean problem and memory
    # finalize(prob)
    # GC.gc(true)

    # # Compute end point as OPF solution
    # eval_f, eval_g!, eval_grad_f!, eval_jac_g!, eval_h!, nv, v_L, v_U, nc,
    #     c_L, c_U, nele_jac, nele_hess, u_0_red, x_0 = OPF.make_ipopt_callbacks(
    #     case_dir, case_qc, tol_inner, tol_pf, iter_max_pf)
    # v0 = vcat(u_0_red, x_0)
    # prob = Ipopt.CreateIpoptProblem(nv, v_L, v_U, nc, c_L, c_U, nele_jac,
    #     nele_hess, eval_f, eval_g!, eval_grad_f!, eval_jac_g!, eval_h!)
    # prob.x = v0
    # # Set Ipopt options
    # Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "exact")
    # Ipopt.AddIpoptNumOption(prob, "tol", tol_inner)
    # Ipopt.AddIpoptIntOption(prob, "max_iter", iter_max_outer)
    # Ipopt.AddIpoptNumOption(prob, "constr_viol_tol", tol_inner)
    # Ipopt.AddIpoptNumOption(prob, "acceptable_tol", tol_inner)
    # Ipopt.AddIpoptNumOption(prob, "acceptable_constr_viol_tol", tol_inner)
    # solve_status = Ipopt.IpoptSolve(prob)
    # if solve_status != 0
    #     error("OPF problem could not be solved.")
    # end
    # u_end = prob.x[1:dim_u]
    # # Clean problem and memory
    # finalize(prob)
    # GC.gc(true)

    # Compute start and end points
    n = case_data.qc_data.n
    mpc_mod = deepcopy(mpc)
    # Instantiate solver (no console output)
    solver = PM.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0,
        "constr_viol_tol" => tol_inner, "acceptable_constr_viol_tol" => tol_inner)
    function get_opf_u(model)
        # TODO: is the rectangular model the fastest?
        # opf_x_dict = PM.solve_opf(model, PM.ACRPowerModel,
        #     solver)["solution"]["bus"]
        # opf_x_dict = PM.solve_model(model, PM.ACRPowerModel,
        #     solver, PM.build_opf)["solution"]["bus"]
        jump_model = PM.instantiate_model(model, PM.ACRPowerModel, PM.build_opf;
            ref_extensions=[])
        opf_x_dict = PM.optimize_model!(jump_model, relax_integrality=false,
            optimizer=solver, solution_processors=[])["solution"]["bus"]
        opf_x = zeros(Float64, 2*n)
        for (bus, vir) in opf_x_dict
            k = parse(Int, bus)
            opf_x[k] = vir["vr"]
            opf_x[n+k] = vir["vi"]
        end
        opf_u = (case_data.J_c + 0.5 .* sum(map((k) ->
            case_data.J_k[k]*opf_x[k], 1:(2*n))))*opf_x
        return opf_u, opf_x
    end
    # Get end point from original OPF solution
    u_end_full, x_end = get_opf_u(mpc_mod)
    # Compute start point as the solution of the minimum loss problem
    # Losses are minimized by minimizing the total generated power
    for (key, _) in mpc_mod["gen"]
        mpc_mod["gen"][key]["cost"] = [0.0; 1.0; 0.0] # Minimum loss Problem
        # mpc_mod["gen"][key]["cost"] .*= -1 # Maximum cost problem
        # mpc_mod["gen"][key]["cost"] = [0.0; randn(rng); 0.0] # Random direction linear cost
        # mpc_mod["gen"][key]["cost"] .*= randn(rng)
        # cost_vec = mpc_mod["gen"][key]["cost"]
        # Q = qr(cost_vec).Q[:,2:3]
        # coeff = randn(rng, 2, 1)
        # coeff ./ norm(coeff, 2)
        # mpc_mod["gen"][key]["cost"] = Q*coeff
    end
    u_start_full, x_start = get_opf_u(mpc_mod)

    return u_start_full, x_start, u_end_full, x_end
end


function load_case(case_dir::String, case_qc::Tuple,
    ind_u::Union{Vector{Int64},UnitRange{Int64},Nothing}=nothing,
    Q::Union{Matrix{Float64},Nothing}=nothing,
    u_center::Union{Vector{Float64},Nothing}=nothing)
    # extract MATPOWER case data
    qc_data, ind, c, opf_u, opf_x, opf_fobj, mpc = case_qc
    n = qc_data.n
    if typeof(ind_u) <: Nothing
        # Set the inputs of PV nodes and the slack as the
        # control variables
        ind_u = vcat(ind.pv, ind.pvd .+ n)
    end
    # Input wrapper for this test case
    PdQd = vcat(qc_data.Pd, qc_data.Qd)
    PdQd[findall(c.vd)] .= 0.0
    PdQd[.+(n,findall(c.pvd))] .= 0.0
    if typeof(Q) <: Nothing || typeof(u_center) <: Nothing
        case_input = (input::Vector{Float64},
            u0::Vector{Float64}) -> begin
            # Input is a translation of selected entries of u
            u = deepcopy(u0)
            u[ind_u] = input - PdQd[ind_u]
            return u
        end::Vector{Float64}
    else
        case_input = (input::Vector{Float64},
            u0::Vector{Float64}) -> begin
            # Input are subspace coordinates
            u = deepcopy(u0)
            u[ind_u] = Q*input + u_center
            return u
        end::Vector{Float64}
    end
    # Power flow solver wrapper
    pf_solver = (input, u0, tol_pf, iter_max_pf, x=nothing, eval=false) ->
        QC.rect_pf(case_input(input, u0), ind.pq, ind.pv, ind.vd,
        qc_data.Yrec_mod, tol_pf, iter_max_pf, x, eval)
    # Compute Jacobian derivatives
    get_jac = (x) -> QC.rect_jacobian(x, ind.pv, ind.vd,
        qc_data.Yrec_mod)
    # TODO: Improve Jacobian computation for OPF constraints
    J_c = get_jac(zeros(Float64, 2*n))
    J_k = fill(SA.spzeros(Float64, 2*n, 2*n), 2*n)
    for k in 1:(2*n) # TODO: parallelize with pmap or whatever
        # Standard basis vector
        e_k = zeros(Float64, 2*n)
        e_k[k] = 1
        J_k[k] = get_jac(e_k) - J_c
    end
    # J0 = J_c + sum(map((k) -> J_k[k]*qc_data.x0[k], 1:(2*n)))
    # fullprint(max(abs.(J0-qc_data.J0)...))

    # Save and return struct
    return TestCase(case_dir, ind_u, Q, u_center, qc_data, ind, c, opf_u,
        opf_x, opf_fobj, case_input, pf_solver, get_jac, J_c, J_k, PdQd, mpc)
end


function load_case(case_dir::String,
    ind_u::Union{Vector{Int64},UnitRange{Int64},Nothing}=nothing,
    Q::Union{Matrix{Float64},Nothing}=nothing,
    u_center::Union{Vector{Float64},Nothing}=nothing)
    # Parse MATPOWER case
    case_qc = QC.compute_qc(case_dir, !true)
    return load_case(case_dir, case_qc, ind_u, Q, u_center)
end


function make_functions(case::TestCase, tol_const::Union{Float64,Vector{Float64}},
    tol_pf=1e-8, iter_max_pf=20, u0=nothing)
    # It is recommended to set tol_const = tol_inner
    # The vector tol_const MUST NEVER include an entry for the power
    # flow feasibility constraint.
    if typeof(u0) <: Nothing
        # No input given
        u0 = case.qc_data.u0
    end
    n = case.qc_data.n
    n_opf_u = length(case.opf_u.c)
    n_opf_x = length(case.opf_x.c)
    n_rate_con = length(case.opf_x.rate_con_buses) # number of line rate constraints
    n_u = length(case.ind_u)
    ind_u = case.ind_u
    opf_u = case.opf_u
    opf_x = case.opf_x
    J_c = case.J_c
    J_k = case.J_k
    Q = case.Q
    u_center = case.u_center
    Hpf = case.qc_data.Hpf

    # Some constraints may be equalities instead, add tolerance
    # to ensure that there are interior points.
    if typeof(tol_const) == Float64
        opf_u_c = opf_u.c .- tol_const
        opf_x_c = opf_x.c .- tol_const
    else
        opf_u_c = opf_u.c - tol_const[1:n_opf_u]
        opf_x_c = opf_x.c - tol_const[(n_opf_u+1):(n_opf_u+n_opf_x)]
    end

    # Feasible set oracle
    df_du = SA.sparse(ind_u, 1:n_u, -1.0, 2*n, n_u)
    function point_oracle(u::Vector{Float64},
        pf_feas::Bool=false,
        x_prev::Union{Vector{Float64},Nothing}=nothing,
        dual_z::Union{Vector{Float64},Nothing}=nothing)
        # Initialize variables
        n_con = n_opf_u + n_opf_x
        if pf_feas
            n_con += 1 # include power flow feasible set
        end
        u_full = case.case_input(u, u0)
        cons = Vector{Float64}(undef, n_con)
        # Power flow and Jacobian
        if typeof(x_prev) <: Nothing || any(isnan.(x_prev))
            # No valid initial guess, use flat start for power flow
            local x, flag = case.pf_solver(u, u0, tol_pf, iter_max_pf)
        else
            # Use provided initial guess for power flow
            local x, flag = case.pf_solver(u, u0, tol_pf, iter_max_pf,
                x_prev)
        end
        if flag == 0
            fullprint("Power flow did not converge.")
            return ([1.0], nothing, nothing, nothing, nothing)
        end
        # return ([-1.0], [], [])
        # Evaluate constraints that depend on u
        cons[1:n_opf_u] = opf_u_c + opf_u.J*u_full
        for j in 1:n_opf_u
            cons[j] += 0.5 * (u_full' * (opf_u.H[j]*u_full))
        end
        # Evaluate constraints that depend on x
        cons[(n_opf_u+1):(n_opf_u+n_opf_x)] = opf_x_c + opf_x.J*x
        for j in 1:n_opf_x
            quad_term = (x' * (opf_x.H[j]*x))/2
            if j <= n_rate_con
                # Line rate, given as a I² limit. We convert it to a S² limit
                # via multiplying by V²
                bus = opf_x.rate_con_buses[j]
                V2 = x[bus]^2 + x[n+bus]^2
                quad_term *= V2
            end
            cons[n_opf_u+j] += quad_term
        end
        # Evaluate pf feasibility constraint
        J_matrix = case.get_jac(x)
        J = lu(J_matrix)
        if pf_feas
            cons[n_con] = -abs(detJ)
        end
        # Jacobian linear system solver function
        J_sol = (x) -> (J.U\(J.L\((J.Rs .* x)[J.p,:]))
            )[invperm(J.q),:]

        # Jacobian of constraints
        Du_gI = zeros(Float64, n_con, n_u)

        # Compute first order derivative of x wrt u
        dx_dut = -J_sol(df_du)

        # Do not compute derivatives if duals were not provided
        if typeof(dual_z) <: Nothing
            return (cons, nothing, nothing, x)
        end

        # Compute Lagrangian Hessian term of constraints that depend on x
        ∂zgX_∂xᵀ = zeros(Float64, 1, 2*n)
        ∂²zgX_∂x² = zeros(Float64, 2*n, 2*n)
        for j in 1:(n_con-n_opf_u)
            if j > n_opf_x
                # Compute derivatives of pf feasible set wrt x
                Jinv_Jk = Array{Matrix{Float64}}(undef, 2*n)
                dg_dx = zeros(Float64, 2*n)
                for k in 1:(2*n)
                    Jinv_Jk[k] = J_sol(J_k[k])
                    dg_dx[k] = -abs(detJ) * tr(Jinv_Jk[k])
                end
                d2g_dx2 = zeros(Float64, 2*n, 2*n)
                for k in 1:(2*n)
                    for i in 1:k
                        d2g_dx2[i,k] = dg_dx[i]*tr(Jinv_Jk[k]) +
                            abs(detJ)*dot(Jinv_Jk[i]', Jinv_Jk[k])
                        d2g_dx2[k,i] = d2g_dx2[i,k]
                    end
                end
            else
                # Compute derivatives of constraint wrt x
                # The Jacobian row should be transposed, but for some reason it
                # returns a column vector directly
                # TODO: can we reuse constraint computations here?
                dg_dx = opf_x.H[j]*x + opf_x.J[j,:]
                if j <= n_rate_con
                    d2g_dx2 = deepcopy(opf_x.H[j])
                    # Correct derivatives of line rates so that they correspond
                    # to S² rates instead of I² rates.
                    bus = opf_x.rate_con_buses[j]
                    I2 = (x' * dg_dx)/2
                    V2 = x[bus]^2 + x[n+bus]^2
                    # We need the gradient of I² to compute the Hessian correction
                    d2g_dx2 .*= V2
                    d2g_dx2[:,bus] .+= (2*x[bus]) * dg_dx
                    d2g_dx2[:,n+bus] .+= (2*x[n+bus]) * dg_dx
                    d2g_dx2[bus,:] .+= (2*x[bus]) * dg_dx
                    d2g_dx2[n+bus,:] .+= (2*x[n+bus]) * dg_dx
                    d2g_dx2[bus,bus] += 2*I2
                    d2g_dx2[n+bus,n+bus] += 2*I2
                    # Now we can correct the gradient
                    dg_dx .*= V2
                    dg_dx[bus] += 2*x[bus]*I2
                    dg_dx[n+bus] += 2*x[n+bus]*I2
                else
                    d2g_dx2 = opf_x.H[j] # no deepcopy needed in this case
                end
            end
            # Compute derivatives of constraints wrt u
            Du_gI[n_opf_u+j,:] = dg_dx' * dx_dut
            ∂zgX_∂xᵀ += dg_dx' .* dual_z[n_opf_u+j]
            # .+= is faster than +=, I guess it leverages sparsity
            ∂²zgX_∂x² .+= d2g_dx2 .* dual_z[n_opf_u+j]
            # axpy!(dual_z[n_opf_u+j], d2g_dx2, ∂²zgX_∂x²) # is not faster
        end
        # Add column correction to second derivative
        θ₁ = (J' \ ∂zgX_∂xᵀ')' #TODO: Optimize this line
        kmax = 2*n
        for k = 1:kmax
            # Even though I'm assigning a row, Julia asks the rhs
            # to be a column vector.
            ∂²zgX_∂x²[k,:] -= (θ₁*J_k[k])'
        end
        # Compute Hessian of Lagrangian term
        D²u_gI = dx_dut' * ∂²zgX_∂x² * dx_dut # TODO: optimize this line

        # Compute derivatives of constraints that depend on u
        # TODO: Check correctness of this
        for j in 1:n_opf_u
            Du_gI[j,:] = (opf_u.H[j]*u_full + opf_u.J[j,:])[ind_u]'
            D²u_gI .+= opf_u.H[j][ind_u,ind_u] .* dual_z[j]
        end

        # Compute derivatives wrt u over subspace, if provided
        if !(typeof(Q) <: Nothing || typeof(u_center) <: Nothing)
            Du_gI = Du_gI * Q
            D²u_gI = Q' * D²u_gI * Q
        end

        return (cons, Du_gI, D²u_gI, x)
    end

    # This function evaluates the oracle over all variable path points
    function path_oracle(u::Array{Vector{Float64}},
        pf_feas::Bool=false,
        x_prev::Union{Array{Vector{Float64}},Nothing}=nothing,
        dual_z::Union{Array{Vector{Float64}},Nothing}=nothing)
        # Create array of nothings if needed
        if typeof(x_prev) <: Nothing
            x_prev = fill(nothing, length(u))
        end
        if typeof(dual_z) <: Nothing
            dual_z = fill(nothing, length(u))
        end
        map_fun = USE_PARALLEL ? pmap : map
        output = map_fun((u,xp,z) -> point_oracle(u, pf_feas, xp, z),
            u, x_prev, dual_z)
        kmax = length(output)
        return Tuple([output[k][i] for k = 1:kmax] for i = 1:4)
    end

    # Objective function
    function f_obj(pk, wk)
        n_points = length(pk)
        sd2k = [wk[k]*norm(pk[k] - pk[k-1])^2 for k = 2:n_points]
        prepend!(sd2k, [0.0])
        output = sd2k[end]
        for k in 2:(n_points-1)
            output += sd2k[k]
        end
        return output
    end

    # Return case functions
    return CaseFunctions(point_oracle, path_oracle, f_obj)
end


function get_feasible_path(case::TestCase, u_start, u_end, tvec,
    tol_outer=1e-3, tol_inner=1e-6, tol_pf=1e-8, iter_max=100,
    iter_max_pf=20, μ_large=1e-5, nu=1e6, u0=nothing, save_hist=false)
    if typeof(u0) <: Nothing
        # No input given
        u0 = case.qc_data.u0
    end
    dim_u = length(u_start)
    n_points = length(tvec)
    path_oracle = make_functions(case, tol_inner, tol_pf,
        iter_max_pf, u0).path_oracle
    η = 1.01

    # Make uniformly space line as starting path
    v0 = vcat(map((t) -> t*u_end + (1.0-t)*u_start, tvec)...)
    # Path vector to array of path points function
    get_pk = (v) -> map((k) -> v[(k*dim_u+1):((k+1)*dim_u)],
        1:(n_points-2)) # exclude extreme points

    # Function to get maximum constraint violation
    eval_path = (v) -> max.(path_oracle(get_pk(v))[1]...)

    # Compute state vector of each point in the path
    con_pk, _, _, x_pk0 = path_oracle(get_pk(v0))

    # Compute maximum violation of starting path
    beta_vec = max.(max.(con_pk...) .+ tol_inner, 0.0)
    # beta_vec .+= 2*tol_inner # force one iteration, TODO: delete
    fullprint("β = $(max(beta_vec...))")
    beta_vec .*= η # leave some gap between path and barrier
    beta_prev = deepcopy(beta_vec)

    # Save iteration data if required
    if save_hist
        v0_hist, beta_hist = deepcopy([[v0]]), deepcopy([beta_vec])
    else
        v0_hist, beta_hist = nothing, nothing
    end

    # Find initial feasible path
    while max(beta_vec...) > tol_inner
        # Solve shortest path problem with large penalty
        case_functions = make_functions(case, beta_vec .+ tol_inner, tol_pf,
            iter_max_pf, u0)
        v0, x_pk0, exit_flag, v_hist = get_shortest_path(case_functions,
            tvec, v0; x_pk0, tol_outer, alpha_min=tol_inner,
            iter_max=iter_max, μ=μ_large, nu, save_hist)

        if exit_flag == 0
            fullprint("Shortest path optimization did not succeed at this step.")
        elseif exit_flag == -1
            @warn "Iteration diverged, feasible path not found."
            break
        end
        # Compute new relaxation parameter vector
        beta_prev = deepcopy(beta_vec)
        beta_vec = max.(eval_path(v0) .+ tol_inner, 0.0)
        beta_vec = min.(beta_vec, beta_prev)
        fullprint("β = $(max(beta_vec...))")
        beta_vec .*= η # leave some gap between path and barrier
        beta_vec = min.(beta_vec, beta_prev) # prevent beta from increasing
        # Save new beta and path, if required
        if save_hist
            append!(v0_hist, [v_hist])
            append!(beta_hist, [beta_vec])
        end
        # Report failure if beta does not decrease
        if max(beta_vec...) > tol_inner &&
            max(abs.(beta_prev - beta_vec)...) <= tol_outer
            # max(abs.(beta_prev - beta_vec)...) <= tol_inner # TODO: Is this correct?
            @warn "Feasible path not found, possibly it does not exist."
            break
        end
    end
    return v0, x_pk0, beta_vec, v0_hist, beta_hist
end


function get_shortest_path(functions::CaseFunctions, tvec, v0; x_pk0=nothing,
    tol_outer=1e-3, alpha_min=1e-6, iter_max=100, μ=1e-6,
    nu=1e6, save_hist=false)
    # Solver parameters
    rho_max = 100.0
    tau = 0.99

    # Compute problem data
    dim_u = Int(length(v0) / length(tvec))
    n_points = length(tvec)
    tdist = tvec[2:end] - tvec[1:(end-1)]
    prepend!(tdist, [0.0])
    wk = map((k) -> 1/((tdist[k]^2)*(n_points - 1)), 2:n_points)
    prepend!(wk, [0.0])
    # append!(wk, [0.0])
    # Unwrap functions
    path_oracle = functions.path_oracle
    f_obj = functions.f_obj # includes barrier term

    # Compute path points and corresponding states
    v = deepcopy(v0)
    pk = map((k) -> v[(k*dim_u+1):((k+1)*dim_u)], 0:(n_points-1))
    if typeof(x_pk0) <: Nothing
        gI, _, _, x_pk_prev = path_oracle(pk)
    else
        x_pk_prev = deepcopy(x_pk0)
        gI = path_oracle(pk, false, x_pk_prev)[1]
    end
    # Compute number of constraints
    n_con = length(gI[1])

    # Equality constraints evaluation function
    ck_fun = (pk) -> begin
        ck = map((k) -> wk[k]*norm(pk[k] - pk[k-1])^2, 2:n_points)
        ck = ck[1:(end-1)] .- ck[2:end]
        return ck
    end

    # Dual variables of equality constraints
    y = zeros(Float64, n_points - 2)
    # Append dummy values at extremes for convenience
    prepend!(y, [0.0])
    append!(y, [0.0])

    # Dual variables of inequality constraints
    z = zeros(Float64, (n_points-2)*n_con)

    # Newton iteration
    c1_armijo = 1e-4
    if any(vcat(gI...) .>= 0)
        @warn "Initial path is not feasible."
        exit_flag = (-1)::Int64 # infeasible starting path
        return nothing, nothing, exit_flag, nothing
    end
    if save_hist
        v_hist = deepcopy([v])
    else
        v_hist = nothing
    end
    err_μ = tol_outer + 1e6
    x_pk = []
    pk_prev = []
    v_prev = []
    y_prev = [] #TODO: Delete maybe?
    z_prev = [] #TODO: Delete maybe?
    correct_inertia = false
    iter = 0
    while iter < iter_max
        dk = map((k) -> pk[k] - pk[k-1], 2:n_points)
        prepend!(dk, [zeros(Float64, dim_u)])

        # Compute equality constraints
        ck = ck_fun(pk)

        # Compute Jacobian of equality constraints
        ∂cj_∂pjm1 = map((k) -> (2*wk[k]) .* dk[k]', 2:n_points)
        ∂cj_∂pj = ∂cj_∂pjm1[1:(end-1)] .+ ∂cj_∂pjm1[2:end]
        # ∂cj_∂pjm1 = -∂cj_∂pjm1[2:(end-1)]
        Dv_cE = BBM.BlockTridiagonal(-∂cj_∂pjm1[2:(end-1)], ∂cj_∂pj,
            -∂cj_∂pjm1[2:(end-1)])

        # Compute gradient of equality constraints Lagrangian term
        ∇v_cE = Dv_cE' * y[2:(end-1)]

        # Compute Hessian blocks of equality constraints Lagrangian term
        ∂ycE_∂pjm1∂pj = map((k) -> Diagonal(2*wk[k]*(y[k] - y[k-1]) * I,
            dim_u), 2:n_points)

        # Compute gradient of objective function
        ∂J_∂pk = map((k) -> 2*(wk[k]*dk[k] - wk[k+1]*dk[k+1]),
            2:(n_points-1))
        ∂J_∂v = vcat(∂J_∂pk...)

        # Build tridiagonal blocks for Hessian of objective function
        ∂²J_∂pkm1∂pk = map((k) -> Diagonal(2*wk[k]*I, dim_u),
            2:n_points)

        # Compute inequality constraints Jacobian, Hessian,
        # Lagrange term gradient and slack variables
        if iter == 0 #TODO: should z be allowed to iterate freely?
            # First iteration, make z = μ ./ s
            gI = path_oracle(pk[2:(end-1)], false, x_pk_prev)[1]
            z_k = [-μ ./ gI[k-2] for k = 3:n_points]
            z = vcat(z_k...)
        else
            z_k = [z[((k-3)*n_con+1):((k-2)*n_con)] for k = 3:n_points]
        end
        gI, Dpk_gI_pk, D²pk_gI_pk, x_pk = path_oracle(pk[2:(end-1)],
            false, x_pk_prev, z_k)
        s = .- gI
        # Compute Lagrangian term gradient
        # This is exactly the gradient of the barrier functions at pk when
        # z = μ ./ s
        ∇pk_gI = transpose.(Dpk_gI_pk) .* z_k

        # Reshape variables to convenience
        s = vcat(s...)
        gI_vec = vcat(gI...)
        ∇v_gI = vcat(∇pk_gI...)
        Dv_gI = BBM.BlockDiagonal(Dpk_gI_pk)

        # Inertia correction
        if correct_inertia
            # TODO: Reuse derivatives from past (failed) iteration
            # TODO: Check correctness
            min_coeff = -min(0, [wk[k]*(y[k] - y[k-1]) for k in 2:n_points]...)
            # D²v_cE += (min_coeff * 4 * (1 + cos(pi/(n_points-1)))) * I
            for k in 1:(n_points-2)
                D²pk_gI_pk[k] += (min_coeff * 4 * (1 + cos(pi/(n_points-1))))*I
                D²pk_gI_pk[k] += norm(D²pk_gI_pk[k], 2)*I
            end
        end

        # Compute gradient of Lagrangian
        ∇v_L = ∂J_∂v + ∇v_cE + ∇v_gI

        # Compute objective function (including barrier)
        if any(gI_vec .>= 0)
            fk = Inf
        else
            fk = f_obj(pk, wk)
            for k in 0:(n_points-3)
                fk += -μ * sum(log.(-gI[k+1]))
            end
        end

        # Check for convergence
        if iter > 0
            rho_d = max(rho_max, (norm(y[2:(end-1)], 1) + norm(z, 1)) /
            (length(y) + length(z) - 2)) / rho_max
            rho_c = max(rho_max, norm(z, 1) / length(z)) / rho_max
            err_μ = max(norm(∇v_L, Inf) / rho_d,
                norm(s .* z .- μ, Inf) / rho_c, norm(ck, Inf))
            if VERBOSE; println([fk err_μ]); end
            if err_μ <= tol_outer
                break
            end
        end

        # Reduced Newton step
        Φ_k = map((k) -> vcat(
            hcat(∂²J_∂pkm1∂pk[k] + ∂ycE_∂pjm1∂pj[k], ∂cj_∂pjm1[k]'),
            hcat(∂cj_∂pjm1[k], 0)),
            1:(n_points-1))
        Σ = Diagonal(z ./ s)
        # Compute step
        D²pk_gI_pk_red = D²pk_gI_pk .+ [Dpk_gI_pk[k]' *
            (Diagonal((z ./ s)[((k-1)*n_con+1):(k*n_con)]) * Dpk_gI_pk[k])
            for k in 1:(n_points-2)]
        D²pk_gI_pk_red = cat.(D²pk_gI_pk_red, 0; dims=(1,2))
        M = BBM.BlockTridiagonal(-Φ_k[2:(end-1)],
            Φ_k[1:(end-1)] .+ Φ_k[2:end] .+ D²pk_gI_pk_red,
            -Φ_k[2:(end-1)])
        ∇v_L_red = ∇v_L + Dv_gI'*(Σ*gI_vec + (μ ./ s))
        Δf = [vcat(∇v_L_red[((k-1)*dim_u+1):(k*dim_u)], ck[k])
            for k in 1:(n_points-2)]
        Δf = vcat(Δf...)
        sol = -(M \ Δf)
        Δv = vcat([sol[((k-1)*(dim_u+1)+1):(k*(dim_u+1)-1)]
            for k in 1:(n_points-2)]...)
        Δy = sol[(dim_u+1):(dim_u+1):end]
        Δz = Σ*(Dv_gI*Δv + gI_vec + (μ ./ z))
        Δpk = map((k) -> Δv[(k*dim_u+1):((k+1)*dim_u)], 0:(n_points-3))
        prepend!(Δpk, [zeros(Float64, dim_u)])

        # Compute step length for dual variables z and y
        ind_Δz⁻ = Δz .< 0.0
        z_step = min(1, ((-tau .* z[ind_Δz⁻]) ./ Δz[ind_Δz⁻])...)
        Δz *= z_step
        Δy *= z_step

        # Armijo condition backtracking
        alpha_k = 1.0
        nu = 0.0 #TODO: delete?
        ϕk = nu * norm(ck, 1) # l1-norm penalty
        dir_fk = 0.0
        for k in 0:(n_points-3)
            dir_fk += Δpk[k+2]' * ∂J_∂pk[k+1]
            dir_fk += Δpk[k+2]' * (Dpk_gI_pk[k+1]' * (-μ ./ gI[k+1]))
        end
        dir_cE = Dv_cE * Δv
        alpha_fun = (alpha) -> begin
            # Generate candidate path
            pk_new = deepcopy(pk)
            for k in 0:(n_points-3)
                pk_new[k+2] = pk[k+2] + alpha .* Δpk[k+2]
            end
            gI_vec_new = vcat(path_oracle(pk_new[2:(end-1)], false,
                x_pk_prev)[1]...)
            if any(gI_vec_new .>= 0)
                return false
            else
                fk_new = f_obj(pk_new, wk) - μ .* sum(log.(-gI_vec_new))
            end
            ϕ_new = nu * norm(ck_fun(pk_new), 1)
            # Instead of differentiating the norm penalty term, we compute
            # the difference of the new penalty value using linearized
            # constrants. For small α this difference converges to the
            # directional derivative times α.
            lin_term = alpha * dir_fk
            lin_term += nu * norm(ck .+ alpha .* dir_cE, 1)
            lin_term -= ϕk
            # Armijo condition
            cond = (fk_new + ϕ_new - fk - ϕk - c1_armijo * lin_term <= 0.0)
            # Add Theorem 1 necessary conditions
            if cond
                bj = map((j) -> wk[j] .* (pk_new[j] - pk_new[j-1]), 2:n_points)
                bj_norm = norm.(bj, Inf)
                cond = min(bj_norm...) > alpha_min * max(bj_norm...)
            end
            if cond
                qj = map((j) -> bj[j] ./ (norm(bj[j], 2)^2), 1:(n_points-1))
                cond = norm(.+(qj...), Inf) / (n_points - 1) >
                    alpha_min * max(norm.(qj, Inf)...)
            end
            return cond
        end

        # Backtracking to find feasible step length
        while alpha_k > alpha_min
            # The conditional always fails for NaN results, as desired
            if alpha_fun(alpha_k)
                break
            else
                alpha_k /= 2.0
            end
        end
        if alpha_k <= alpha_min
            if correct_inertia
                fullprint("Line search failed after inertia correction, " *
                    "terminating iteration.")
                break
            else
                fullprint("Line search failed, repeating step with inertia correction.")
                correct_inertia = true
                continue
            end
        end
        # fullprint(alpha_min)

        # Line search succeeded, no inertia correction needed for next iteration
        correct_inertia = false

        # Update path
        for k in 0:(n_points-3)
            dpk = Δv[(k*dim_u+1):((k+1)*dim_u)]
            pk[k+2] = pk[k+2] + alpha_k .* dpk
        end
        # .+= throws an error, better avoid it
        z += alpha_k .* Δz
        y[2:(end-1)] += alpha_k .* Δy
        v = vcat(pk...)

        # Advance iteration counter
        iter += 1
        pk_prev = deepcopy(pk)
        v_prev = deepcopy(v)
        x_pk_prev = deepcopy(x_pk)

        # Store iteration data, if required
        if save_hist
            append!(v_hist, [v])
        end
    end

    # Report convergence
    exit_flag = 0::Int64 # did not converge
    if err_μ <= tol_outer
        exit_flag = 1::Int64 # attained convergence
    end

    return v, x_pk, exit_flag, v_hist
end

end