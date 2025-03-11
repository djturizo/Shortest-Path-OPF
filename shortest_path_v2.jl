module ShortestPathOPF

using LinearAlgebra
import PowerModels as PM, SparseArrays as SA
import Ipopt as Ipopt
import BlockArrays as BA, BlockBandedMatrices as BBM
# import JSON
import ForwardDiff as FD

# using Infiltrator
# Infiltrator.toggle_async_check(false)
# Infiltrator.clear_disabled!()

include("power_flow_rect_qc.jl")
import .PowerFlowRectQC as QC

# include("interior_point_solver.jl")
# import .InteriorPointSolver as IPS
# include("opf_wrapper.jl")
# import .OPFWrapper as OPF

global USE_PARALLEL = true
const VERBOSE = !true


function fullprint(var)
    if VERBOSE
        show(stdout, "text/plain", var)
        println()
    end
    return
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
    opf_u_H_sym
    opf_x_H_sym
    Hpf_sym
end

mutable struct CaseFunctions
    case
    point_oracle
    path_oracle
    f_obj
    dim_u
    n
    n_opf_u
    n_opf_x
    opf_u_c
    opf_x_c
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
        case_input = (input,
            u0::AbstractVector) -> begin
            # Input is a translation of selected entries of u
            n_u0 = length(u0)
            u = Vector{eltype(input)}(undef, n_u0)
            u[:] = u0
            u[ind_u] = input - PdQd[ind_u]
            return u
        end
    else
        case_input = (input,
            u0::AbstractVector) -> begin
            # Input are subspace coordinates
            u = deepcopy(u0)
            u[ind_u] = Q*input + u_center
            return u
        end
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

    # The Hessians are symmetric, so we make new objects that store only the lower triangular
    # part. This reduces the cost of symmetric Hessian operations like matrix-scalar products
    # and matrix additions. Matrix-vector products still required the full Hessian.
    n_opf_u = length(opf_u.c)
    n_opf_x = length(opf_x.c)
    opf_u_H_sym = Array{AbstractMatrix}(undef, n_opf_u)
    opf_x_H_sym = Array{AbstractMatrix}(undef, n_opf_x)
    Hpf_sym = Array{AbstractMatrix}(undef, 2*n)
    for j in 1:n_opf_u
        opf_u_H_sym[j] = tril(opf_u.H[j][ind_u,ind_u])
    end
    for j in 1:n_opf_x
        opf_x_H_sym[j] = tril(opf_x.H[j])
    end
    jmax = 2*n
    for j in 1:jmax
        Hpf_sym[j] = tril(qc_data.Hpf[j])
    end

    # Save and return struct
    return TestCase(case_dir, ind_u, Q, u_center, qc_data, ind, c, opf_u,
        opf_x, opf_fobj, case_input, pf_solver, get_jac, J_c, J_k, PdQd, mpc,
        opf_u_H_sym, opf_x_H_sym, Hpf_sym)
end


function load_case(case_dir::String,
    ind_u::Union{Vector{Int64},UnitRange{Int64},Nothing}=nothing,
    Q::Union{Matrix{Float64},Nothing}=nothing,
    u_center::Union{Vector{Float64},Nothing}=nothing)
    # Parse MATPOWER case
    case_qc = QC.compute_qc(case_dir, !true)
    return load_case(case_dir, case_qc, ind_u, Q, u_center)
end


function relax_constraints!(functions::CaseFunctions,
    tol_const::Union{Float64,Vector{Float64}})
    # Some constraints may be equalities instead, add tolerance
    # to ensure that there are interior points.
    opf_u_c = functions.case.opf_u.c
    opf_x_c = functions.case.opf_x.c
    n_opf_u = functions.n_opf_u
    n_opf_x = functions.n_opf_x
    if typeof(tol_const) == Float64
        functions.opf_u_c = opf_u_c .- tol_const
        functions.opf_x_c = opf_x_c .- tol_const
    else
        functions.opf_u_c = opf_u_c - tol_const[1:n_opf_u]
        functions.opf_x_c = opf_x_c - tol_const[(n_opf_u+1):(n_opf_u+n_opf_x)]
    end
    return functions
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
    if typeof(case.Q) <: Nothing
        Q = nothing
    else
        Q = SA.sparse(case.Q)
    end
    n = case.qc_data.n
    n_opf_u = length(case.opf_u.c)
    n_opf_x = length(case.opf_x.c)
    n_rate_con = length(case.opf_x.rate_con_buses) # number of line rate constraints
    n_u = length(case.ind_u)
    ind_u = case.ind_u
    opf_u = case.opf_u
    opf_x = case.opf_x
    # J_c = case.J_c
    # J_k = case.J_k
    u_center = case.u_center
    # Hpf = case.qc_data.Hpf
    opf_u_H_sym = case.opf_u_H_sym
    opf_x_H_sym = case.opf_x_H_sym
    Hpf_sym = case.Hpf_sym

    # Feasible set oracle
    df_du = SA.sparse(ind_u, 1:n_u, -1.0, 2*n, n_u)
    function point_oracle(functions::CaseFunctions, u::AbstractArray{T}, x::AbstractArray{T},
        dual_pf::Union{AbstractVector{Float64},Nothing}=nothing,
        dual_z::Union{AbstractVector{Float64},Nothing}=nothing;
        first_iter::Bool=false, μ=nothing) where T
        # Initialize variables
        n_con = n_opf_u + n_opf_x
        u_full = case.case_input(u, u0)
        cons = Vector{T}(undef, n_con)
        Hu = Vector{Vector{T}}(undef, n_opf_u)
        Hx = Vector{Vector{T}}(undef, n_opf_x)
        # Evaluate constraints that depend on u
        cons[1:n_opf_u] = functions.opf_u_c + opf_u.J*u_full
        for j in 1:n_opf_u
            Hu[j] = opf_u.H[j]*u_full
            cons[j] += 0.5 * (u_full' * Hu[j])
        end
        # Evaluate constraints that depend on x
        cons[(n_opf_u+1):(n_opf_u+n_opf_x)] = functions.opf_x_c + opf_x.J*x
        for j in 1:n_opf_x
            Hx[j] = opf_x.H[j]*x
            quad_term = (x' * Hx[j])/2
            if j <= n_rate_con
                # Line rate, given as a I² limit. We convert it to a S² limit
                # via multiplying by V²
                bus = opf_x.rate_con_buses[j]
                V2 = x[bus]^2 + x[n+bus]^2
                quad_term *= V2
            end
            cons[n_opf_u+j] += quad_term
        end

        # Evaluate power flow constraints
        pf_cons = case.pf_solver(u, u0, tol_pf, iter_max_pf, x, true)[1]

        # Do not compute derivatives if duals were not provided and this is not the first
        # iteration
        if !first_iter && typeof(dual_z) <: Nothing
            return (cons, pf_cons,
                nothing, nothing, nothing, nothing, nothing, nothing, nothing)
        end

        # If this is the first iteration we compute the derivatives using default values of
        # the dual variables. In this case the barrier parameter 'μ' must have been provided
        if first_iter
            if isnothing(μ)
                error("Barrier parameter must be provided in order to generate default " *
                    "values of the dual variables")
            end
            dual_pf = zeros(eltype(T), 2*n)
            dual_z = (-μ) ./ cons
        end

        # Jacobian of constraints
        # Compute the inequality Jacobians transposed and then transpose at then. This is
        # cheaper.
        Du_cE = copy(df_du)
        Dx_cE = case.get_jac(x)

        # Hessian of constraints (D²u_cE is always the zero matrix)
        D²u_gI = SA.spzeros(Float64, n_u, n_u)
        # 'D²x_gI_pf' includes both inequality terms and power flow terms
        D²x_gI_pf = SA.spzeros(Float64, 2*n, 2*n)

        # Compute Hessian of power flow terms in Lagrangian
        n_x = 2*n
        for j = 1:n_x
            D²x_gI_pf += Hpf_sym[j] * dual_pf[j]
        end

        # Compute derivatives of constraints that depend on u.
        # TODO: Check correctness of this
        nzval = Vector{Vector{Float64}}(undef, n_opf_u)
        rowval = Vector{Vector{Int64}}(undef, n_opf_u)
        colptr = ones(Int64, n_opf_u+1)
        for j in 1:n_opf_u
            # Transposed Jacobian column
            col_j = (Hu[j] + opf_u.J[j,:])[ind_u]
            # Hessian
            D²u_gI += opf_u_H_sym[j] * dual_z[j]
            # Store transposed Jacobian column
            nzval[j] = col_j.nzval
            rowval[j] = col_j.nzind
            colptr[j+1] = colptr[j] + length(col_j.nzval)
        end
        Dᵗu_gI = SA.SparseMatrixCSC{Float64,Int64}(n_u, n_opf_u,
            colptr, vcat(rowval...), vcat(nzval...))

        # Compute derivatives of constraints that depend on x
        nzval = Vector{Vector{Float64}}(undef, n_opf_x)
        rowval = Vector{Vector{Int64}}(undef, n_opf_x)
        colptr = ones(Int64, n_opf_x+1)
        for j in 1:n_opf_x
            dg_dx = Hx[j] + opf_x.J[j,:]
            if j <= n_rate_con
                d2g_dx2 = copy(opf_x.H[j])
                # Correct derivatives of line rates so that they correspond
                # to S² rates instead of I² rates.
                bus = opf_x.rate_con_buses[j]
                I2 = (x' * dg_dx)/2 # line rate cons. do not have linear term
                V2 = x[bus]^2 + x[n+bus]^2
                # We need the gradient of I² to compute the Hessian correction
                # TODO: is it worth it to only perform these operations on the lower
                # triangular part of the Hessian?
                d2g_dx2 .*= V2
                d2g_dx2[:,bus] .+= (2*x[bus]) * dg_dx
                d2g_dx2[:,n+bus] .+= (2*x[n+bus]) * dg_dx
                d2g_dx2[bus,:] .+= (2*x[bus]) * dg_dx
                d2g_dx2[n+bus,:] .+= (2*x[n+bus]) * dg_dx
                d2g_dx2[bus,bus] += 2*I2
                d2g_dx2[n+bus,n+bus] += 2*I2
                # The Hessian is symmetric, so we only need the lower triangular part
                d2g_dx2 = tril(d2g_dx2)
                # Now we can correct the gradient
                dg_dx .*= V2
                dg_dx[bus] += 2*x[bus]*I2
                dg_dx[n+bus] += 2*x[n+bus]*I2
            else
                d2g_dx2 = opf_x_H_sym[j]
            end
            # Update the Hessian
            D²x_gI_pf += d2g_dx2 * dual_z[n_opf_u+j]
            # Store transposed Jacobian column
            nzval[j] = dg_dx.nzval
            rowval[j] = dg_dx.nzind
            colptr[j+1] = colptr[j] + length(dg_dx.nzval)
        end
        Dᵗx_gI = SA.SparseMatrixCSC{Float64,Int64}(2*n, n_opf_x,
            colptr, vcat(rowval...), vcat(nzval...))

        # Compute derivatives wrt u over subspace, if provided
        if !(typeof(Q) <: Nothing || typeof(u_center) <: Nothing)
            # Du_gI = Du_gI * Q
            Dᵗu_gI = Q' * Dᵗu_gI
            Du_cE = Du_cE * Q
            D²u_gI_diag = SA.spdiagm(diag(D²u_gI))
            D²u_gI = tril(Q' * (D²u_gI + D²u_gI' - D²u_gI_diag) * Q)
        end

        # WARN: We only store the lower triangular part of 'D²u_gI' and 'D²x_gI_pf', as the
        # Hessians are symmetric
        if first_iter
            return (cons, pf_cons, Dᵗu_gI', Dᵗx_gI', Du_cE, Dx_cE, D²u_gI, D²x_gI_pf,
                dual_pf, dual_z)
        end
        return (cons, pf_cons, Dᵗu_gI', Dᵗx_gI', Du_cE, Dx_cE, D²u_gI, D²x_gI_pf)
    end

    # This function evaluates the oracle over all variable path points
    function path_oracle(functions::CaseFunctions, p::AbstractArray{T},
        dual_pf::Union{AbstractArray{T},Nothing}=nothing,
        dual_z::Union{AbstractArray{T},Nothing}=nothing;
        first_iter::Bool=false, μ=nothing) where {T <: AbstractVector}
        # Create array of nothings if needed
        if typeof(dual_pf) <: Nothing
            dual_pf = fill(nothing, length(p))
        end
        if typeof(dual_z) <: Nothing
            dual_z = fill(nothing, length(p))
        end
        dim_p = length(p[1])
        dim_u = dim_p - 2*n
        kmax = length(p)
        oracle_fun = (p,d_pf,z) -> point_oracle(functions, view(p, 1:dim_u),
            view(p, (dim_u+1):dim_p), d_pf, z; first_iter, μ)
        # map_fun = USE_PARALLEL ? pmap : map
        # output = map_fun(oracle_fun, p, dual_pf, dual_z)
        output = Array{Any}(undef, kmax)
        if USE_PARALLEL
            Threads.@threads :greedy for k = 1:kmax
                output[k] = oracle_fun(p[k], dual_pf[k], dual_z[k])
            end
        else
            for k = 1:kmax
                output[k] = oracle_fun(p[k], dual_pf[k], dual_z[k])
            end
        end
        if first_iter
            return Tuple([output[k][i] for k = 1:kmax] for i = 1:10)
        else
            return Tuple([output[k][i] for k = 1:kmax] for i = 1:8)
        end
    end

    # Objective function
    function f_obj(pk, wk)
        n_points = length(pk)
        sd2k = [wk[k]*norm(view(pk[k], 1:n_u) - view(pk[k-1], 1:n_u), 2)^2 for k = 2:n_points]
        prepend!(sd2k, [0.0])
        output = sd2k[end]
        kmax = n_points - 1
        for k in 2:kmax
            output += sd2k[k]
        end
        return output/kmax
    end

    out = CaseFunctions(case, point_oracle, path_oracle, f_obj, n_u, n, n_opf_u, n_opf_x,
        opf_u.c, opf_x.c)
    return relax_constraints!(out, tol_const)
end


function get_straight_line_path(case::TestCase, u_start, u_end, tvec, tol_pf=1e-8,
    iter_max_pf=20, u0=nothing)
    if typeof(u0) <: Nothing
        # No input given
        u0 = case.qc_data.u0
    end
    n = case.qc_data.n
    dim_u = length(u_start)
    dim_p = dim_u + 2*n;
    n_points = length(tvec)

    # Make uniformly spaced line in the space of u as starting path
    p_start = vcat(u_start, zeros(Float64, 2*n))
    p_end = vcat(u_end, zeros(Float64, 2*n))
    v0 = vcat(map((t) -> t*p_end + (1.0-t)*p_start, tvec)...) # missing components in x
    # Path vector to array of path points function
    get_pk = (v) -> map((k) -> view(v, (k*dim_p+1):((k+1)*dim_p)),
        1:(n_points-2)) # exclude extreme points
    pk_0 = get_pk(v0)
    # We still need to compute the x components of the extreme points, so we will include
    # in 'pk_0'
    prepend!(pk_0, [view(v0, 1:dim_p)])
    append!(pk_0, [view(v0, ((n_points-1)*dim_p+1):(n_points*dim_p))])

    # PF computation function
    get_x = (k) -> begin
        local x, flag = case.pf_solver(pk_0[k][1:dim_u], u0, tol_pf, iter_max_pf)
        if flag == 0
            error("Power flow did not converge at one of the points on the initial path.")
        end
        return x
    end

    # Solve PF at each point
    x_k = Array{Vector{eltype(pk_0[1])}}(undef, n_points)
    if USE_PARALLEL
        Threads.@threads :greedy for k = 1:n_points; x_k[k] = get_x(k); end
    else
        for k = 1:n_points; x_k[k] = get_x(k); end
    end
    for k = 1:n_points
        # The entries of 'pk_0' are views, so this assignment modifies v0 directly
        pk_0[k][(dim_u+1):end] = x_k[k]
    end

    return pk_0, v0
end


function get_feasible_path(case::TestCase, u_start, u_end, tvec,
    tol_outer=1e-3, tol_inner=1e-6, tol_pf=1e-8, iter_max=100,
    iter_max_pf=20, μ_large=1e-2, nu_0=1e-6, u0=nothing, save_hist=false)
    if typeof(u0) <: Nothing
        # No input given
        u0 = case.qc_data.u0
    end
    n = case.qc_data.n
    dim_u = length(u_start)
    dim_p = dim_u + 2*n;
    n_points = length(tvec)
    case_functions = make_functions(case, 0.0, tol_pf, iter_max_pf, u0)
    path_oracle = case_functions.path_oracle
    η = 1.01
    tol_beta = 0.001

    # Make uniformly spaced line in the space of u as starting path. The components in x are
    # are determined by solving the power flow
    pk_0, v0 = get_straight_line_path(case, u_start, u_end, tvec, tol_pf, iter_max_pf, u0)

    # Path vector to array of path points function
    get_pk = (v) -> map((k) -> view(v, (k*dim_p+1):((k+1)*dim_p)),
        1:(n_points-2)) # exclude extreme points

    # Function to get maximum constraint violation
    eval_path = (v) -> max.(path_oracle(case_functions, get_pk(v))[1]...)

    # Compute state vector of each point in the path
    con_pk = path_oracle(case_functions, pk_0)[1]

    # Compute maximum violation of starting path
    beta = maximum(vcat(con_pk...))
    fullprint("β = $(beta)")
    beta_prev = deepcopy(beta)

    # Save iteration data if required
    if save_hist
        v0_hist, beta_hist = deepcopy([[v0]]), deepcopy([beta])
    else
        v0_hist, beta_hist = nothing, nothing
    end

    # Find initial feasible path
    case_functions_relaxed = make_functions(case, η*beta, tol_pf,
        iter_max_pf, u0)
    path_data = nothing
    total_iter = 0
    while beta >= tol_inner
        # Previous path data is available to reuse. We must shift the values of gI (stored in
        # 'path_data[1]') to account for the new relaxation parameter
        if !isnothing(path_data) && length(path_data) > 2
            kmax = n_points - 2
            for k = 1:kmax
                path_data[1][k] .+= η*(beta_prev - beta)
            end
        end

        # Solve shortest path problem with large penalty
        beta_decrease = (1 - tol_beta) * beta
        v0, exit_flag, v_hist, path_data, iter = get_shortest_path(case_functions_relaxed,
            tvec, v0; tol_outer, iter_max=iter_max, μ=μ_large, nu_0, save_hist,
            beta_decrease, beta=(η*beta), path_data)
        total_iter += iter
        if exit_flag == 0
            fullprint("Shortest path optimization did not succeed at this step.")
        elseif exit_flag == -1
            @warn "Iteration diverged, feasible path not found."
            break
        end
        # Compute new relaxation parameter
        beta_prev = deepcopy(beta)
        beta = maximum(eval_path(v0))
        fullprint("β = $(beta)")
        # Save new beta and path, if required
        if save_hist
            append!(v0_hist, [v_hist])
            append!(beta_hist, [beta])
        end
        # Report failure if beta does not decrease
        rel_decrease = (beta_prev - beta) / beta_prev
        if beta >= tol_inner &&
            # # if the barrier problem converged we can admit decreases smaller than
            # # 'tol_outer', but they still have to be larger than 'tol_outer'.
            # ((exit_flag == 1 && rel_decrease <= tol_outer) ||
            # # If the barrier problem didn't converge we can only accept decreases larger
            # # than 'tol_beta'
            # (exit_flag != 1 && exit_flag != 2 && rel_decrease <= tol_beta))
            rel_decrease <= tol_beta
            @warn "Feasible path not found, possibly it does not exist."
            break
        end
        # Update relaxed constraints
        relax_constraints!(case_functions_relaxed, η*beta)
    end

    # Previous path data is available to reuse. We must shift the values of gI (stored in
    # 'path_data[1]') to obtain the unrelaxed constraint values
    if !isnothing(path_data) && length(path_data) > 2
        kmax = n_points - 2
        for k = 1:kmax
            path_data[1][k] .+= η*beta_prev
        end
    end

    # return output data
    return v0, beta, v0_hist, beta_hist, path_data, total_iter
end


function get_shortest_path(functions::CaseFunctions, tvec, v0;
    tol_outer=1e-3, alpha_min=1e-2, tol_comp=1e-6, iter_max=100, μ=1e-6,
    nu_0=1e-6, save_hist=false, beta_decrease=-Inf, beta=nothing, path_data=nothing)
    # Solver parameters
    c1_armijo = 1e-4
    rho_max = 100.0
    tau = 0.99
    κ_nu = 0.1
    # nu = 1.0e6
    nu = nu_0
    penalize_path_cons = true
    short_step_limit = 6 + Inf

    # Compute path points
    n = functions.n
    dim_u = functions.dim_u
    dim_p = dim_u + 2*n
    n_points = length(tvec)
    v = deepcopy(v0)
    pk = map((k) -> v[(k*dim_p+1):((k+1)*dim_p)], 0:(n_points-1))

    # Compute problem data
    n_opf_u = functions.n_opf_u
    n_opf_x = functions.n_opf_x
    n_con = n_opf_u + n_opf_x
    tdist = tvec[2:end] - tvec[1:(end-1)]
    prepend!(tdist, [0.0])
    endpoint_dist2 = norm(pk[end][1:dim_u] - pk[1][1:dim_u], 2)^2
    wk = map((k) -> 1/((tdist[k]^2)*endpoint_dist2), 2:n_points)
    # wk = map((k) -> 1/(tdist[k]^2), 2:n_points)
    prepend!(wk, [0.0])
    # append!(wk, [0.0])
    δ_S = (1e-4)*norm(wk, 2) / (n_points - 1)
    # Unwrap functions
    path_oracle = functions.path_oracle
    f_obj = functions.f_obj # doesn't include barrier term

    # Compute path constraints
    if isnothing(path_data)
        gI, cE_pf, Du_gI, Dx_gI, Du_cE_pf, Dx_cE_pf, D²u_gI, D²x_gI_pf, dual_pf, z =
            path_oracle(functions, view(pk, 2:(n_points-1)); first_iter=true, μ=μ)
        dual_pf = vcat(dual_pf...)
        z = vcat(z...)
    elseif length(path_data) == 2
        dual_pf, z = path_data
        z_k_temp = [z[((k-1)*n_con+1):(k*n_con)] for k = 1:(n_points-2)]
        dual_pf_k_temp = [dual_pf[((k-1)*2*n+1):(k*2*n)] for k = 1:(n_points-2)]
        gI, cE_pf, Du_gI, Dx_gI, Du_cE_pf, Dx_cE_pf, D²u_gI, D²x_gI_pf =
            path_oracle(functions, view(pk, 2:(n_points-1)), dual_pf_k_temp, z_k_temp)
    else
        gI, cE_pf, Du_gI, Dx_gI, Du_cE_pf, Dx_cE_pf, D²u_gI, D²x_gI_pf, dual_pf, z =
            path_data
    end

    # Path equality constraints evaluation function
    ck_path_fun = (pk) -> begin
        local ck = map((k) -> wk[k]*norm(view(pk[k], 1:dim_u) - view(pk[k-1], 1:dim_u))^2,
            2:n_points)
        return ck[1:(end-1)] .- ck[2:end]
    end

    # Dual variables of path equality constraints
    y = zeros(Float64, n_points - 2)
    # Append dummy values at extremes for convenience
    prepend!(y, [0.0])
    append!(y, [0.0])

    # # Dual variables of power flow equality constraints
    # dual_pf = zeros(Float64, (n_points-2)*(2*n))

    # # Dual variables of inequality constraints
    # z = zeros(Float64, (n_points-2)*n_con)

    # Declare vars that can be reused when repeating a step, so that they are not gc'ed after
    # a loop iteration.
    ∇²v_J, ∇²v_ycE, ∇v_L, s, cE_pf_vec, ck, gI_vec, Dv_cE, Dv_gI, ∂J_∂pk, fk, ∂²J_∂pj∂pj,
        ∂²ycE_∂pj∂pj, z_k, s_k, Dpk_gI, ∂cj_∂pj, ∂cj_∂pjm1, ∂²J_∂pjm1∂pj, ∂²ycE_∂pjm1∂pj =
        (nothing for _ = 1:20)

    # Newton iteration
    exit_flag = 0::Int64
    if any(vcat(gI...) .>= 0)
        fullprint(maximum(vcat(gI...)))
        @warn "Initial path is not feasible."
        exit_flag = (-1)::Int64 # infeasible starting path
        return nothing, exit_flag, nothing, nothing, 0
    end
    if save_hist
        v_hist = deepcopy([v])
    else
        v_hist = nothing
    end
    err_μ = Inf
    v_prev = []
    correct_inertia = false
    short_step_count = 0
    repeat_step = false
    iter = 0
    while iter < iter_max
        if !repeat_step
            z_k = [z[((k-1)*n_con+1):(k*n_con)] for k = 1:(n_points-2)]
            dual_pf_k = [dual_pf[((k-1)*2*n+1):(k*2*n)] for k = 1:(n_points-2)]
            if iter > 0
                gI, cE_pf, Du_gI, Dx_gI, Du_cE_pf, Dx_cE_pf, D²u_gI, D²x_gI_pf =
                    path_oracle(functions, view(pk, 2:(n_points-1)), dual_pf_k, z_k)
            end
            s_k = .- gI

            # Reshape variables to convenience
            s = vcat(s_k...)
            gI_vec = vcat(gI...)

            # If all constraints are less than the specified violation limit beta, return
            # the current path
            if beta_decrease > -Inf && maximum(gI_vec) + beta < beta_decrease
                path_data = (gI, cE_pf, Du_gI, Dx_gI, Du_cE_pf, Dx_cE_pf, D²u_gI, D²x_gI_pf,
                    dual_pf, z)
                return v, 2, v_hist, path_data, iter
            end

            # Compute Lagrangian term gradient
            # This is exactly the gradient of the barrier functions at pk when z = μ ./ s
            ∇pk_gI = [vcat(Du_gI[k]' * view(z_k[k], 1:n_opf_u),
                Dx_gI[k]' * view(z_k[k], (n_opf_u+1):n_con)) for k = 1:(n_points-2)]

            # Reshape variables to convenience
            cE_pf_vec = vcat(cE_pf...)
            ∇v_gI = vcat(∇pk_gI...)
            # 'blockdiag' doesn't work with adjoint objects, so I gotta use 'cat' instead
            Dpk_gI = [cat(Du_gI[k], Dx_gI[k]; dims=(1,2)) for k = 1:(n_points-2)]
            Dv_gI = SA.blockdiag(Dpk_gI...)

            # Compute consecutive corner differences over path
            dk = map((k) -> vcat(view(pk[k], 1:dim_u) - view(pk[k-1], 1:dim_u),
                SA.spzeros(Float64, 2*n)), 2:n_points)
            prepend!(dk, [zeros(Float64, dim_p)])

            # Compute equality constraints
            ck = ck_path_fun(pk)

            # Compute Jacobian of path equality constraints
            ∂cj_∂pjm1 = map((k) -> (2*wk[k]) * dk[k]', 2:n_points)
            ∂cj_∂pj = ∂cj_∂pjm1[1:(end-1)] + ∂cj_∂pjm1[2:end]
            ∂cj_∂pjm1 = ∂cj_∂pjm1[2:(end-1)]
            if isempty(∂cj_∂pjm1)
                Dv_cE_subdiag = SA.spzeros(0, 0)
            else
                # 'blockdiag' doesn't work with adjoint objects, so I gotta use 'cat' instead
                Dv_cE_subdiag = cat(∂cj_∂pjm1...; dims=(1,2))
            end
            Dv_cE = cat(∂cj_∂pj...; dims=(1,2)) -
                    [SA.spzeros(1, (n_points-2)*dim_p);
                    Dv_cE_subdiag SA.spzeros((n_points-3), dim_p)] -
                    [SA.spzeros((n_points-3), dim_p) Dv_cE_subdiag;
                    SA.spzeros(1, (n_points-2)*dim_p)]

            # Compute gradient of path equality constraints' Lagrangian term
            ∇v_cE = Dv_cE' * y[2:(end-1)]

            # Compute Hessian blocks of path equality constraints' Lagrangian term. We compute
            # only the lower triangular part, because the Hessian is symmetric
            ∂²ycE_∂pjm1∂pj = map((k) -> SA.spdiagm(vcat(fill(2*wk[k]*(y[k] - y[k-1]), dim_u),
                zeros(Float64, 2*n))), 2:n_points)
            ∂²ycE_∂pj∂pj = ∂²ycE_∂pjm1∂pj[1:(end-1)] + ∂²ycE_∂pjm1∂pj[2:end]
            ∂²ycE_∂pjm1∂pj = ∂²ycE_∂pjm1∂pj[2:(end-1)]
            ∇²v_ycE_subdiag = SA.blockdiag(∂²ycE_∂pjm1∂pj...)
            ∇²v_ycE = SA.blockdiag(∂²ycE_∂pj∂pj...) -
                    [SA.spzeros(dim_p, (n_points-2)*dim_p);
                    ∇²v_ycE_subdiag SA.spzeros((n_points-3)*dim_p, dim_p)]

            # Compute gradient of objective function
            ∂J_∂pk = map((k) -> (2/(n_points-1))*(wk[k]*dk[k] - wk[k+1]*dk[k+1]),
                2:(n_points-1))
            ∂J_∂v = vcat(∂J_∂pk...)

            # Build tridiagonal blocks for Hessian of objective function. We compute
            # only the lower triangular part, because the Hessian is symmetric
            ∂²J_∂pjm1∂pj = map((k) -> SA.spdiagm(vcat(fill((2/(n_points-1))*wk[k], dim_u),
                zeros(Float64, 2*n))), 2:n_points)
            ∂²J_∂pj∂pj = ∂²J_∂pjm1∂pj[1:(end-1)] + ∂²J_∂pjm1∂pj[2:end]
            ∂²J_∂pjm1∂pj = ∂²J_∂pjm1∂pj[2:(end-1)]
            ∇²v_J_subdiag = SA.blockdiag(∂²J_∂pjm1∂pj...)
            ∇²v_J = SA.blockdiag(∂²J_∂pj∂pj...) -
                    [SA.spzeros(dim_p, (n_points-2)*dim_p);
                    ∇²v_J_subdiag SA.spzeros((n_points-3)*dim_p, dim_p)]

            # Compute gradient of Lagrangian
            ∇pk_cE_pf = [vcat(Du_cE_pf[k]' * dual_pf_k[k], Dx_cE_pf[k]' * dual_pf_k[k])
                for k = 1:(n_points-2)]
            ∇v_L = ∂J_∂v + ∇v_gI + ∇v_cE + vcat(∇pk_cE_pf...)

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
            rho_d = max(rho_max, (norm(y[2:(end-1)], 1) + norm(dual_pf, 1) + norm(z, 1)) /
                (length(y) - 2 + length(dual_pf) + length(z))) / rho_max
            rho_c = max(rho_max, norm(z, 1) / length(z)) / rho_max
            err_μ = max(norm(∇v_L, Inf) / rho_d,
                norm(s .* z .- μ, Inf) / rho_c, norm(cE_pf_vec, Inf), norm(ck, Inf))
            if VERBOSE; println([fk err_μ]); end
            # if VERBOSE; println([norm(∇v_L, Inf), norm(s .* z .- μ, Inf),
            #     norm(cE_pf_vec, Inf), norm(ck, Inf)]); end
            if err_μ <= tol_outer
                exit_flag = 1::Int64
                break
            end
        end

        # Inertia correction
        inertia_corrected = false
        if correct_inertia
            # TODO: Check correctness
            min_coeff = -min(0, [wk[k]*(y[k] - y[k-1]) for k in 2:n_points]...)
            for k in 1:(n_points-2)
                D²u_gI[k] += ((min_coeff * 4 * (1 + cos(pi/(n_points-1)))) +
                    norm(D²u_gI[k], 2) + δ_S)*I
                D²x_gI_pf[k] += (norm(D²x_gI_pf[k], 2) + δ_S)*I
            end
            correct_inertia = false
            inertia_corrected = true
        # else
        #     # Correct for decoupling the Hessians
        #     for k in 1:(n_points-2)
        #         coeff = (2/(n_points-1))*(wk[k+1] + wk[k+2]) + wk[k+1]*abs(y[k+1] - y[k])
        #             + wk[k+2]*abs(y[k+2] - y[k+1])
        #         D²u_gI[k] += coeff*I
        #     end
        end

        # # Full Newton step
        # # First we compute the Hessian
        # ∇²pk_L = [SA.blockdiag(D²u_gI[k], D²x_gI_pf[k]) for k = 1:(n_points-2)]
        # ∇²v_L = ∇²v_J + ∇²v_ycE + SA.blockdiag(∇²pk_L...)
        # # Compute Jacobian of full set of equality constraints
        # Dpk_cE_pf = [hcat(Du_cE_pf[k], Dx_cE_pf[k]) for k = 1:(n_points-2)]
        # Dv_cE_pf = SA.blockdiag(Dpk_cE_pf...)
        # # Build mismatch vector for Newton step
        # Δf = [∇v_L; z - μ ./ s; cE_pf_vec; ck; gI_vec + s]
        # # Build Newton matrix and compute step
        # Σ = SA.spdiagm(z ./ s)
        # M1 = SA.blockdiag(∇²v_L, Σ)
        # M2 = [Dv_cE_pf SA.spzeros(size(Dv_cE_pf, 1), size(Σ, 2));
        #       Dv_cE    SA.spzeros(size(Dv_cE, 1), size(Σ, 2));
        #       Dv_gI    SA.sparse(I, size(Dv_gI, 1), size(Dv_gI, 1))]
        # # M = Symmetric(hcat([M1; M2], SA.spzeros(size(M1, 1) + size(M2, 1),
        # #     size(M2, 1))), :L)
        # # sol = -(M \ Vector(Δf)) # no sparse LdL' solver yet in Julia
        # # sol = -(M \ Δf)
        # M1_diag = SA.spdiagm(diag(M1))
        # M = [M1 + M1' - M1_diag  M2';
        #      M2                  SA.spzeros(size(M2, 1), size(M2, 1))]
        # sol = -(lu(M) \ Vector(Δf))
        # # Split step to the parts of each variable
        # ind_start, ind_end = 1, dim_p*(n_points-2)
        # Δv = sol[ind_start:ind_end]
        # ind_start, ind_end = ind_end+1, ind_end+(n_con*(n_points-2))
        # Δs = sol[ind_start:ind_end]
        # ind_start, ind_end = ind_end+1, ind_end+(2*n*(n_points-2))
        # Δdual_pf = sol[ind_start:ind_end]
        # ind_start, ind_end = ind_end+1, ind_end+(n_points-2)
        # Δy = sol[ind_start:ind_end]
        # ind_start, ind_end = ind_end+1, ind_end+(n_con*(n_points-2))
        # Δz = sol[ind_start:ind_end]

        # Full Newton step
        # Build Newton matrix in block tridiagonal form. A_k holds the diagonal blocks
        ∇²pk_L = [SA.blockdiag(D²u_gI[k], D²x_gI_pf[k]) for k = 1:(n_points-2)]
        Dpk_cE_pf = [hcat(Du_cE_pf[k], Dx_cE_pf[k]) for k = 1:(n_points-2)]
        Dv_cE_pf = SA.blockdiag(Dpk_cE_pf...)
        A_k = [[SA.blockdiag(∂²J_∂pj∂pj[k] + ∂²ycE_∂pj∂pj[k] + ∇²pk_L[k],
                    SA.spdiagm(z_k[k] ./ s_k[k]));
                Dpk_cE_pf[k] SA.spzeros(2*n, n_con);
                Dpk_gI[k]    SA.sparse(I, n_con, n_con)]
                for k = 1:(n_points-2)]
        A_k = [hcat(A_k[k], SA.spzeros(size(A_k[k], 1), 2*n + n_con))
            for k = 1:(n_points-2)]
        A_k = [[[∂cj_∂pj[k]'; SA.spzeros(n_con + 2*n + n_con, 1)] A_k[k]]
                for k = 1:(n_points-2)]
        A_k = [[SA.spzeros(1, dim_p + n_con + 2*n + n_con + 1);
                A_k[k]] for k = 1:(n_points-2)]
        # B_k holds the off-diagonal blocks
        B_k = [[SA.spzeros(1, 1) -∂cj_∂pjm1[k];
                -∂cj_∂pjm1[k]'   -(∂²J_∂pjm1∂pj[k] + ∂²ycE_∂pjm1∂pj[k])]
                # # decouple the Hessians
                # -∂cj_∂pjm1[k]'   -0*(∂²J_∂pjm1∂pj[k] + ∂²ycE_∂pjm1∂pj[k])]
                for k = 1:(n_points-3)]
        B_k = [[B_k[k] SA.spzeros(1 + dim_p, n_con + 2*n + n_con);
                SA.spzeros(n_con + 2*n + n_con, 1 + dim_p + n_con + 2*n + n_con)]
                for k = 1:(n_points-3)]
        # Build lower triangular part of the full Newton matrix M. Recall that for the
        # diagonal blocks A_k we're only storing the lower triangular part.
        dim_block = size(A_k[1], 1)
        M = SA.blockdiag(A_k...) +
            [SA.spzeros(dim_block, dim_block*(n_points-2));
             SA.blockdiag(B_k...) SA.spzeros(dim_block*(n_points-3), dim_block)]
        # Build mismatch vector for Newton step
        ∇pk_L = map((k) -> ∇v_L[(k*dim_p+1):((k+1)*dim_p)], 0:(n_points-3))
        Δf_k = [[ck[k]; ∇pk_L[k]; z_k[k] - μ ./ s_k[k]; cE_pf[k]; gI[k] + s_k[k]]
            for k = 1:(n_points-2)]
        # Build Newton matrix and compute step
        M = M + M' - SA.spdiagm(diag(M))
        # sol = -(Symmetric(M) \ vcat(Δf_k...)) # no sparse symmetric solver yet in Julia
        sol = -(lu(M) \ Vector(vcat(Δf_k...)))
        # Split step to the parts of each variable
        ind_start, ind_end = 1, 1
        Δy = vcat([sol[(k*dim_block+ind_start):(k*dim_block+ind_end)]
            for k = 0:(n_points-3)]...)
        ind_start, ind_end = ind_end+1, ind_end+dim_p
        Δv = vcat([sol[(k*dim_block+ind_start):(k*dim_block+ind_end)]
            for k = 0:(n_points-3)]...)
        ind_start, ind_end = ind_end+1, ind_end+n_con
        Δs = vcat([sol[(k*dim_block+ind_start):(k*dim_block+ind_end)]
            for k = 0:(n_points-3)]...)
        ind_start, ind_end = ind_end+1, ind_end+2*n
        Δdual_pf = vcat([sol[(k*dim_block+ind_start):(k*dim_block+ind_end)]
            for k = 0:(n_points-3)]...)
        ind_start, ind_end = ind_end+1, ind_end+n_con
        Δz = vcat([sol[(k*dim_block+ind_start):(k*dim_block+ind_end)]
            for k = 0:(n_points-3)]...)

        # Compute step length for primal variables s and v (and pk)
        ind_Δs⁻ = Δs .< 0.0
        s_step = min(1, ((-tau .* s[ind_Δs⁻]) ./ Δs[ind_Δs⁻])...)
        Δs *= s_step
        Δv *= s_step
        Δpk = map((k) -> Δv[(k*dim_p+1):((k+1)*dim_p)], 0:(n_points-3))
        Δsk = map((k) -> Δs[(k*n_con+1):((k+1)*n_con)], 0:(n_points-3))

        # Compute step length for dual variables z and y
        ind_Δz⁻ = Δz .< 0.0
        z_step = min(1, ((-tau .* z[ind_Δz⁻]) ./ Δz[ind_Δz⁻])...)
        Δz *= z_step
        Δdual_pf *= z_step
        Δy *= z_step

        # Armijo condition backtracking
        alpha_k = 1.0
        dir_fk = 0.0
        for k in 0:(n_points-3)
            dir_fk += Δpk[k+1]' * ∂J_∂pk[k+1]
            dir_fk += Δsk[k+1]' * (-μ ./ s_k[k+1])
        end
        dir_cE_pf = Dv_cE_pf * Δv
        dir_cE_path = Dv_cE * Δv
        # Update nu
        c_pred = s_step * (norm(cE_pf_vec, 1) + norm(ck, 1)) # predicted constraint decrease
        nu_new = dir_fk / ((1 - κ_nu) * c_pred)
        if nu_new > nu; nu = max(nu_new, 2*nu); end
        ϕk = nu * norm(cE_pf_vec, 1) # penalty term
        if penalize_path_cons; ϕk += nu * norm(ck, 1); end
        # Armijo condition function evaluator
        alpha_fun = (alpha, c_lin) -> begin
            # Generate candidate path
            pk_new = deepcopy(pk)
            for k in 0:(n_points-3)
                pk_new[k+2] = pk[k+2] + (alpha * Δpk[k+1])
            end
            gI_vec_new, cE_pf_new = path_oracle(functions, view(pk_new, 2:(n_points-1)))
            gI_vec_new = vcat(gI_vec_new...)
            cE_pf_new = vcat(cE_pf_new...)
            if any(gI_vec_new .>= 0)
                return Inf
            else
                fk_new = f_obj(pk_new, wk) - μ .* sum(log.(-gI_vec_new))
            end
            ϕ_new = nu * norm(cE_pf_new, 1)
            if penalize_path_cons; ϕ_new += nu * norm(ck_path_fun(pk_new), 1); end
            # Instead of differentiating the norm penalty term, we compute
            # the difference of the new penalty value using linearized
            # constrants. For small α this difference converges to the
            # directional derivative times α.
            lin_term = alpha * dir_fk
            lin_term += nu * norm(cE_pf_vec .+ alpha .* dir_cE_pf, 1)
            if penalize_path_cons; lin_term += nu * norm(ck .+ alpha .* dir_cE_path, 1); end
            lin_term -= ϕk
            # Armijo condition
            gap = fk_new + ϕ_new - fk - ϕk - c_lin * lin_term
            # Add Theorem 1 necessary conditions
            if gap <= 0
                bj = map((j) -> wk[j] .* (view(pk_new[j], 1:dim_u) -
                    view(pk_new[j-1], 1:dim_u)), 2:n_points)
                bj_norm = norm.(bj, Inf)
                cond = min(bj_norm...) > tol_comp * max(bj_norm...)
            end
            if gap <= 0
                qj = map((j) -> bj[j] ./ (norm(bj[j], 2)^2), 1:(n_points-1))
                cond = norm(.+(qj...), Inf) / (n_points - 1) >
                    tol_comp * max(norm.(qj, Inf)...)
            end
            return gap
        end

        # Backtracking to find feasible step length
        repeat_step = false
        while alpha_k > alpha_min
            # The conditional always fails for NaN results, as desired
            if alpha_fun(alpha_k, c1_armijo) <= 0
                break
            else
                alpha_k /= 2
            end
        end
        if alpha_k <= alpha_min
            if inertia_corrected
                fullprint("Line search failed after inertia correction, " *
                    "terminating iteration.")
                break
            else
                fullprint("Line search failed, repeating step with inertia correction.")
                repeat_step = true
                correct_inertia = true
                short_step_count = 0
                continue
            end
        end
        if VERBOSE; fullprint(alpha_k); end

        # Update count of consecutive short steps, and defined if next step will have inertia
        # correction.
        short_step_count = (alpha_k < 1) ? (short_step_count + 1) : 0
        correct_inertia = (short_step_count >= short_step_limit)
        if correct_inertia
            fullprint("Too many consecutive short steps, next step will be computed " *
                "with inertia correction.")
            short_step_count = 0
        end

        # Update path
        for k in 0:(n_points-3)
            pk[k+2] += alpha_k * Δpk[k+1]
        end
        z += alpha_k * Δz
        dual_pf += alpha_k * Δdual_pf
        y[2:(end-1)] += alpha_k * Δy
        v = vcat(pk...)

        # Advance iteration counter
        iter += 1
        v_prev = deepcopy(v)

        # Store iteration data, if required
        if save_hist
            append!(v_hist, [v])
        end
    end

    # Return ourput data
    path_data = (dual_pf, z)
    return v, exit_flag, v_hist, path_data, iter
end

end