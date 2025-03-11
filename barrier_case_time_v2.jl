using LinearAlgebra, Random, StatProfilerHTML, Profile, LaTeXStrings, Printf

import PowerModels as PM, Memento, Ipopt as Ipopt, Logging
import Plots as plt, ColorSchemes as CS

# using Infiltrator
# Infiltrator.toggle_async_check(false)
# # Infiltrator.clear_disabled!()

include("power_flow_rect_qc.jl")
import .PowerFlowRectQC as QC
include("shortest_path_v2.jl")
import .ShortestPathOPF as SP

AUTO_RUN = false; # Flag to run a test case automatically on loading


function run_case(case_dir::String;
    u_start=nothing, u_end=nothing,
    μ=1e-5, μ_large=0.0125, K=9, rng::MersenneTwister=MersenneTwister(),
    print_str=false, print_K=false, disable_msgs=false)
    # Parameters
    gamma = 0.1
    tol_outer = 1e-3
    tol_inner = 1e-6
    tol_eig = 1e-12 # Tolerance for eigenvalue operations
    # tol_svd = 1e-2 # Tolerance for SVD rank cutoff
    iter_max_outer = 1000
    #iter_max_outer = 200
    iter_max_inner = 100
    tol_pf = 1e-8
    iter_max_pf = 20
    nu_0 = 1e-6

    # Disable logging msgs except errors
    if disable_msgs; Logging.disable_logging(Logging.Warn); end

    # Load test case
    local mpc, case_qc
    Memento.setlevel!(PM._LOGGER, "error") do # disable PowerModels warnings temporarily
        mpc = QC.load_case(case_dir)
        case_qc = QC.compute_qc(mpc, !true)
    end
    if case_dir == "MATPOWER/case9dongchan.m"
        # 2D problem for case9-dongchan version
        ind_u = 2:3
        case_data = SP.load_case(case_dir, case_qc, ind_u)
    else
        case_data = SP.load_case(case_dir, case_qc)
    end
    qc_data = case_data.qc_data
    n = qc_data.n
    ind_u = case_data.ind_u
    dim_u = length(ind_u)
    dim_p = dim_u + 2*n
    u0 = qc_data.u0
    PdQd = case_data.PdQd

    # Get start point as the solution of the minimum loss problem,
    # and end point as solution of the OPF.
    if isnothing(u_start) && isnothing(u_end)
        if case_dir == "MATPOWER/case9dongchan.m"
            # Extreme points for case9-dongchan version
            # println(u0[ind_u])
            u_start = deepcopy(u0)
            u_start[2:3] = [0.5; 0.5]
            u_start = u_start[ind_u]
            u_end = deepcopy(u0)
            u_end[2:3] = [1.5; 1.3]
            u_end = u_end[ind_u]
            @QC.fullprint u_start
            @QC.fullprint u_end
            x_start = case_data.pf_solver(u_start, u0, tol_pf, iter_max_pf)[1]
            x_end = case_data.pf_solver(u_end, u0, tol_pf, iter_max_pf)[1]
        else
            u_start_full, x_start, u_end_full, x_end = SP.get_endpoints(
            mpc, case_data, tol_inner, rng)
            # Correct endpoints so as to not include loads
            u_start = u_start_full[ind_u] + PdQd[ind_u]
            u_end = u_end_full[ind_u] + PdQd[ind_u]
            # New base case is the start point
            u0 = deepcopy(u_start_full)
        end
        # println(max(abs.(u_start_full-u_end_full)...))
        # u_diff = [u_start_full[1:end] u_end_full[1:end]][
        #     setdiff(1:end,ind_u),:]
        # @QC.fullprint max(abs.(u_diff[:,1] - u_diff[:,2])...)
        # println("")
    elseif isnothing(u_start) || isnothing(u_end)
        error("Only on of the two endpoints was provided.")
    else
        x_start = case_data.pf_solver(u_start, u0, tol_pf, iter_max_pf)[1]
        x_end = case_data.pf_solver(u_end, u0, tol_pf, iter_max_pf)[1]
    end

    # Problem Parameters
    tspan = (0.0, 1.0)
    dt = 1 / (K + 1)
    tvec = collect(tspan[1]:dt:tspan[2])
    tvec[end] = tspan[2]
    n_points = length(tvec)
    tdist = tvec[2:end] - tvec[1:(end-1)]
    prepend!(tdist, [0.0])

    # Check feasibility of endpoints
    case_functions = SP.make_functions(case_data, tol_inner, tol_pf,
        iter_max_pf, u0)
    oracle = case_functions.point_oracle
    path_oracle = case_functions.path_oracle
    if max(oracle(case_functions, u_start, x_start)[1]...) > 0.0
        @warn "Start point is not feasible"
    end
    if max(oracle(case_functions, u_end, x_end)[1]...) > 0.0
        @warn "End point is not feasible"
    end

    # Profile.clear()
    # Profile.init(n=10^8, delay=0.001)
    # @profile begin
    # try
    GC.gc(true)
    t_start = time()
    #--------------------------------------------------------------------
    # Find initial feasible path
    get_pk = (v) -> map((k) -> v[(k*dim_p+1):((k+1)*dim_p)],
        0:(n_points-1)) # does NOT exclude extreme points
    get_cons_pk = (v) -> map((cons) -> max(cons...),
        path_oracle(case_functions, get_pk(v))[1])
    v0, beta, v0_hist, _, path_data, total_iter = SP.get_feasible_path(case_data, u_start,
        u_end, tvec, tol_outer, tol_inner, tol_pf, iter_max_inner,
        iter_max_pf, μ_large, nu_0, u0, true)
    n_v0 = length(v0_hist)

    exit_flag = nothing
    found_feasible = (beta < tol_inner)
    if beta < tol_inner && !isnothing(path_data)
        # Previous path data is available to reuse. We must shift the values of gI (stored in
        # 'path_data[1]') to account for the relaxation parameter used for computing the
        # shortest path (that is, 'tol_inner').
        if length(path_data) > 2
            kmax = n_points - 2
            for k = 1:kmax
                path_data[1][k] .-= tol_inner
            end
        end

        # Solve shortest path problem
        v, exit_flag, v_hist, _, iter = SP.get_shortest_path(case_functions,
            tvec, v0; tol_outer,
            iter_max=iter_max_inner, μ, nu_0, save_hist=true, path_data)
        total_iter += iter
        found_feasible = true
    else
        # If 'beta < tol_inner' then the straight line is feasible. Otherwise a feasible path
        # was not found. In either case we set to 'v' the last processed path ,'v0'.
        v = v0
    end
    #--------------------------------------------------------------------
    t_end = time()
    # catch; end
    # end
    # statprofilehtml(path="time_test")
    # Profile.clear()
    # return

    # Compute path metrics
    p = 2
    pk=get_pk(v)
    tvec_v = cumsum([norm((pk[i+1] - pk[i])[1:dim_u], p) for i in 1:(length(pk)-1)])
    tvec_v ./= tvec_v[end]
    tvec_v[end] = 1.0
    prepend!(tvec_v, 0.0)
    pk0_0 = SP.get_straight_line_path(case_data, u_start, u_end, tvec, tol_pf,
        iter_max_pf, u0)[1]
    pl0 = sum([norm((pk0_0[i+1] - pk0_0[i])[1:dim_u], p) for i in 1:(length(pk0_0)-1)])
    pl = sum([norm((pk[i+1] - pk[i])[1:dim_u], p) for i in 1:(length(pk)-1)])
    pl_diff = sum([norm(((pk[i+1] - pk[i]) - (pk0_0[i+1] - pk0_0[i]))[1:dim_u], p)
        for i in 1:(length(pk)-1)])
    if beta >= tol_inner
        # Feasible path not found, distance metrics make no sense
        pl0 = NaN
        pl = NaN
        pl_diff = NaN
    end

    # μ_range = (10.0 .^ (-12:0.05:-2))
    # pl_vals = fill(NaN, size(μ_range))
    # for k in eachindex(μ_range)
    #     v, exit_flag = SP.get_shortest_path(case_functions,
    #         tvec, v0; tol_outer,
    #         iter_max=iter_max_inner, μ=μ_range[k], nu_0, path_data)
    #     pl = NaN
    #     if beta < tol_inner
    #         # Compute path metrics
    #         pk=get_pk(v)
    #         tvec_v = cumsum([norm((pk[i+1] - pk[i])[1:dim_u], p) for i in 1:(length(pk)-1)])
    #         tvec_v ./= tvec_v[end]
    #         tvec_v[end] = 1.0
    #         prepend!(tvec_v, 0.0)
    #         pk0_0 = SP.get_straight_line_path(case_data, u_start, u_end, tvec, tol_pf,
    #             iter_max_pf, u0)[1]
    #         pl0 = sum([norm((pk0_0[i+1] - pk0_0[i])[1:dim_u], p)
    #             for i in 1:(length(pk0_0)-1)])
    #         pl = sum([norm(((pk[i+1] - pk[i]) - (pk0_0[i+1] - pk0_0[i]))[1:dim_u], p)
    #             for i in 1:(length(pk)-1)])
    #     end
    #     pl_vals[k] = 100*pl/pl0
    #     GC.gc(true)
    # end
    # # Plot result
    # plt.gr(size=(160*3,120*3), legend=:none)
    # p1 = plt.plot(μ_range, pl_vals, xaxis=:log10, xticks = 10.0 .^ (-12:2:-2),
    #     linewidth=1.5)
    # plt.xlabel!(L"\mu_\textrm{lo}")
    # plt.ylabel!("Extra path length [%]")
    # plt.xflip!(true)
    # plt.display(p1)
    # plt.savefig("pathlength_plot_v2.pdf")

    max_violation = max(max.(path_oracle(case_functions, pk0_0[2:(n_points-1)])[1]...)...)
    max_violation_after = max(max.(path_oracle(case_functions, pk[2:(n_points-1)])[1]...)...)
    path_diff_pc = 100*pl_diff/pl0
    fobj_diff_pc = 100*(pl-pl0)/pl0

    if print_str
        n_gen = Int(round((length(ind_u)+1)/2)) # number of generators
        case_name = join(split(split(case_dir, "/")[end], ".")[1:(end-1)])
        case_name = replace(case_name, "pglib_opf_" => "")
        case_name = replace(case_name, "_" => "\\_")
        if print_K
            @printf "\t&%d\t&%d" K total_iter
        else
            @printf "\\textrm{"
            print(case_name)
            @printf "}"
            @printf "\t&%d\t&%d" n n_gen
        end
        if found_feasible
            test_str = @sprintf "%.2f" path_diff_pc
            prec_1 = (length(test_str) == 4) ? 2 : 1
            test_str = @sprintf "%.2f" fobj_diff_pc
            prec_2 = (length(test_str) == 4) ? 2 : 1
            if exit_flag == 0
                str = @sprintf("\t&%.2E\t&%.1f\t&No*\t&%.2E\t&%.*f\\%%\t&%.*f\\%%\t",
                    max_violation, t_end - t_start, max_violation_after, prec_1, path_diff_pc,
                    prec_2, fobj_diff_pc)
            else
                str = @sprintf("\t&%.2E\t&%.1f\t&Yes\t&%.2E\t&%.*f\\%%\t&%.*f\\%%\t",
                    max_violation, t_end - t_start, max_violation_after, prec_1, path_diff_pc,
                    prec_2, fobj_diff_pc)
            end
        else
            str = @sprintf("\t&%.2E\t&%.1f\t&No\t&%.2E\t&-\t&-\t", max_violation,
                t_end - t_start, max_violation_after)
        end
        str = replace(str, "E-0" => "E-")
        str = replace(str, "E+0" => "E+")
        print(str)
        if print_K
            @printf " \\\\\n\\cline{2-9}\n"
        else
            @printf " \\\\\n\\hline\n"
        end
    end

    # Re-enable logging msgs
    if disable_msgs; Logging.disable_logging(Logging.BelowMinLevel); end

    GC.gc(true);
    return max_violation, max_violation_after, path_diff_pc, fobj_diff_pc, (t_end - t_start)
end

# Run automatically
if AUTO_RUN
    # Set up rng with fixed seed for reproducibility
    # This seed gives "disconected" endpoints for case9mod
    # This seed gives a non-trival path for case39_epri
    # This seed gives "disconected" endpoints for case60_c
    # This seed gives a non-trival path for case73_ieee
    # rng = MersenneTwister();
    rng = MersenneTwister(UInt32[0x2070c972, 0x521b92e3, 0xaf734195, 0x223eab32]);
    # This seed gives possibly connected endpoints for case9mod
    # rng = MersenneTwister(UInt32[0x739de873, 0xd5c78c8a, 0x0403c802, 0xd34c6068])

    # List of test cases to solve_model
    case_dir_vec = String[]

    # Load test cases
    push!(case_dir_vec, "MATPOWER/case9mod.m");
    # push!(case_dir_vec, "MATPOWER/case9dongchan.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case14_ieee.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case24_ieee_rts.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case30_ieee.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case39_epri.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case57_ieee.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case60_c.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case73_ieee_rts.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case89_pegase.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case118_ieee.m");
    # # The cases below run slow
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case162_ieee_dtc.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case200_activ.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case240_pserc.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case300_ieee.m");
    # push!(case_dir_vec, "MATPOWER/pglib_opf_case500_goc.m");

    # Molzahn's difficult cases
    # push!(case_dir_vec, "Molzahn_cases/nmwc3acyclic_connected_feasible_space.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc3acyclic_disconnected_feasible_space.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc3cyclic.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc4.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc5.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc14.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc24.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc57.m");
    # push!(case_dir_vec, "Molzahn_cases/nmwc118.m"); # seems to fail generating endpoints

    # redirect_stdout(devnull) do # suppress prints
    #     # warm-up execution, this speeds up subsequent runs
    #     run_case(case_dir_vec[1]; output_file="plot.pdf", rng=copy(rng));
    # end
    print_K = true
    for case_dir in case_dir_vec
        # for n_segments = 2 .^ (1:7)
        for n_segments = 10:10
            run_case(case_dir; rng=copy(rng), K=n_segments-1, print_str=true,
                print_K, disable_msgs=true);
        end
    end
    return
end