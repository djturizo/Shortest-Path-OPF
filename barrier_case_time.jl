using LinearAlgebra, Random, Suppressor, StatProfilerHTML, Profile, LaTeXStrings

import PowerModels as PM, Ipopt as Ipopt
import Plots as plt, ColorSchemes as CS

using Infiltrator
Infiltrator.toggle_async_check(false)
# Infiltrator.clear_disabled!()

include("power_flow_rect_qc.jl")
import .PowerFlowRectQC as QC
include("shortest_path.jl")
import .ShortestPathOPF as SP

AUTO_RUN = !false; # Flag to run a test case automatically on loading


function run_case(case_dir::String; output_file::String="plot_v1.pdf",
    rng::MersenneTwister=MersenneTwister(), μ=1e-6, μ_large=0.1)
    # Parameters
    gamma = 0.1
    tol_outer = 1e-3
    tol_inner = 1e-6
    tol_eig = 1e-12 # Tolerance for eigenvalue operations
    # tol_svd = 1e-2 # Tolerance for SVD rank cutoff
    iter_max_outer = 1000
    #iter_max_outer = 200
    iter_max_inner = 100
    iter_max_feasible = 20
    tol_pf = 1e-8
    iter_max_pf = 20
    nu = 1e6


    # Load test case
    mpc = QC.load_case(case_dir)
    case_qc = QC.compute_qc(mpc, !true)
    # # ----------------------------TODO: Delete---------------------
    # 2D problem for case9-dongchan version
    ind_u = 2:3
    case_data = SP.load_case(case_dir, case_qc, ind_u)
    # #--------------------------------------------------------------
    # case_data = SP.load_case(case_dir, case_qc)
    #--------------------------------------------------------------
    qc_data = case_data.qc_data
    n = qc_data.n
    ind_u = case_data.ind_u
    dim_u = length(ind_u)
    u0 = qc_data.u0
    PdQd = case_data.PdQd

    # # Get start point as the solution of the minimum loss problem,
    # # and end point as solution of the OPF.
    # u_start_full, x_start, u_end_full, x_end = SP.get_endpoints(
    #     mpc, case_data, tol_inner, rng)
    # # Correct endpoints so as to not include loads
    # u_start = u_start_full[ind_u] + PdQd[ind_u]
    # u_end = u_end_full[ind_u] + PdQd[ind_u]
    # # ------------------------TODO: Delete-------------------------
    # Extreme points for case9-dongchan version
    println(u0[ind_u])
    u_start = deepcopy(u0)
    u_start[2:3] = [0.5; 0.5]
    u_start = u_start[ind_u]
    u_end = deepcopy(u0)
    u_end[2:3] = [1.5; 1.3]
    u_end = u_end[ind_u]
    # #--------------------------------------------------------------
    # # New base case is the start point
    # u0 = deepcopy(u_start_full)
    # println(max(abs.(u_start_full-u_end_full)...))
    # u_diff = [u_start_full[1:end] u_end_full[1:end]][
    #     setdiff(1:end,ind_u),:]
    # @QC.fullprint max(abs.(u_diff[:,1] - u_diff[:,2])...)
    # println("")
    # GC.gc(true) # force GC to clean any Ipopt residues
    #--------------------------------------------------------------

    # Problem Parameters
    tspan = (0.0, 1.0)
    dt = 0.05
    # dt = 0.2
    tvec = collect(tspan[1]:dt:tspan[2])
    tvec[end] = tspan[2]
    n_points = length(tvec)
    tdist = tvec[2:end] - tvec[1:(end-1)]
    prepend!(tdist, [0.0])

    # Check feasibility of endpoints
    SP.open_parallel()
    case_functions = SP.make_functions(case_data, tol_inner, tol_pf,
        iter_max_pf, u0)
    oracle = case_functions.point_oracle
    path_oracle = case_functions.path_oracle
    if max(oracle(u_start)[1]...) > 0.0
        @warn "Start point is not feasible"
    end
    if max(oracle(u_end)[1]...) > 0.0
        @warn "End point is not feasible"
    end
    # @QC.fullprint findmax(oracle(u_start)[1])
    # return

    # Profile.clear()
    # Profile.init(n=10^8, delay=0.01)
    # @profile begin
    # try
    t_start = time()
    #--------------------------------------------------------------------
    # Find initial feasible path
    get_pk = (v) -> map((k) -> v[(k*dim_u+1):((k+1)*dim_u)],
        0:(n_points-1)) # does NOT exclude extreme points
    get_cons_pk = (v) -> map((cons) -> max(cons...), path_oracle(get_pk(v))[1])
    v0, x_pk0, beta_vec, v0_hist, _ = SP.get_feasible_path(case_data, u_start,
        u_end, tvec, tol_outer, tol_inner, tol_pf, iter_max_feasible,
        iter_max_pf, μ_large, nu, u0, true)
    n_v0 = length(v0_hist)

    if !(max(beta_vec...) > tol_inner)
        # Solve shortest path problem
        v, x_pk, exit_flag, v_hist = SP.get_shortest_path(case_functions,
            tvec, v0; x_pk0, tol_outer, alpha_min=tol_inner,
            iter_max=iter_max_inner, μ, nu, save_hist=true)
    else
        v = v0
    end
    #--------------------------------------------------------------------
    t_end = time()
    # catch; end
    # end
    # statprofilehtml(path="time_test")
    # Profile.clear()
    # return

    p = 2
    pk=get_pk(v)
    tvec_v = cumsum([norm(pk[i+1] - pk[i], p) for i in 1:(length(pk)-1)])
    tvec_v ./= tvec_v[end]
    tvec_v[end] = 1.0
    prepend!(tvec_v, 0.0)
    v0_0 = vcat(map((t) -> t*u_end + (1.0-t)*u_start, tvec_v)...)
    pk0_0 = get_pk(v0_0)
    pl0 = sum([norm(pk0_0[i+1]-pk0_0[i], p) for i in 1:(length(pk0_0)-1)])
    pl = sum([norm(pk[i+1] - pk[i], p) for i in 1:(length(pk)-1)])
    pl_diff = sum([norm((pk[i+1]-pk[i]) - (pk0_0[i+1]-pk0_0[i]), p)
        for i in 1:(length(pk)-1)])
    if max(beta_vec...) > tol_inner
        # Feasible path not found, distance metrics make no sense
        pl0 = NaN
        pl = NaN
        pl_diff = NaN
    end

    # μ_range = 10.0 .^ (-11:0.05:-1)
    # pl_vals = fill(NaN, size(μ_range))
    # for k in eachindex(μ_range)
    #     v, x_pk, exit_flag = SP.get_shortest_path(case_functions,
    #             tvec, v0; x_pk0, tol_outer, alpha_min=tol_inner,
    #             iter_max=iter_max_inner, μ=μ_range[k], nu, save_hist=false)
    #     pl = NaN
    #     if !(max(beta_vec...) > tol_inner)
    #         # Compute path metrics
    #         pk=get_pk(v)
    #         tvec_v = cumsum([norm(pk[i+1] - pk[i], p) for i in 1:(length(pk)-1)])
    #         tvec_v ./= tvec_v[end]
    #         tvec_v[end] = 1.0
    #         prepend!(tvec_v, 0.0)
    #         v0_0 = vcat(map((t) -> t*u_end + (1.0-t)*u_start, tvec_v)...)
    #         pk0_0 = get_pk(v0_0)
    #         pl0 = sum([norm(pk0_0[i+1]-pk0_0[i], p) for i in 1:(length(pk0_0)-1)])
    #         pl = sum([norm((pk[i+1]-pk[i]) - (pk0_0[i+1]-pk0_0[i]), p)
    #             for i in 1:(length(pk)-1)])
    #     end
    #     pl_vals[k] = 100*pl/pl0
    #     GC.gc(true)
    # end
    # # Plot result
    # plt.gr(size=(160*3,120*3), legend=:none)
    # p1 = plt.plot(μ_range, pl_vals, xaxis=:log10, xticks = 10.0 .^ (-11:2:-1),
    #     linewidth=1.5)
    # plt.xlabel!(L"\mu_\textrm{lo}")
    # plt.ylabel!("Extra path length [%]")
    # plt.xflip!(true)
    # plt.display(p1)
    # plt.savefig("pathlength_plot.pdf")

    SP.close_parallel()
    max_violation = max(max.(path_oracle(pk0_0[2:(n_points-1)])[1]...)...)
    max_violation_after = max(max.(path_oracle(pk[2:(n_points-1)])[1]...)...)
    n_gen = Int(round((length(ind_u)+1)/2)) # number of generators
    # return n_gen, max_violation, t_end - t_start, pl0, pl, 100*(pl - pl0)/pl0
    return max_violation, max_violation_after, t_end - t_start, 100*pl_diff/pl0,
        100*(pl-pl0)/pl0
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
    # Load test case
    # case_dir = "MATPOWER/case9mod.m";
    case_dir = "MATPOWER/case9dongchan.m";
    # case_dir = "MATPOWER/pglib_opf_case14_ieee.m";
    # case_dir = "MATPOWER/pglib_opf_case24_ieee_rts.m";
    # case_dir = "MATPOWER/pglib_opf_case30_ieee.m";
    # case_dir = "MATPOWER/pglib_opf_case39_epri.m";
    # case_dir = "MATPOWER/pglib_opf_case57_ieee.m";
    # case_dir = "MATPOWER/pglib_opf_case60_c.m";
    # case_dir = "MATPOWER/pglib_opf_case73_ieee_rts.m";
    # case_dir = "MATPOWER/pglib_opf_case89_pegase.m";
    # case_dir = "MATPOWER/pglib_opf_case118_ieee.m";
    # The cases below run slow
    # case_dir = "MATPOWER/pglib_opf_case162_ieee_dtc.m"; #TODO: fix power flow?
    # case_dir = "MATPOWER/pglib_opf_case200_activ.m"; #TODO: fix power flow?

    # Molzahn's difficult cases
    # case_dir = "Molzahn_cases/nmwc3acyclic_connected_feasible_space.m";
    # case_dir = "Molzahn_cases/nmwc3acyclic_disconnected_feasible_space.m";
    # case_dir = "Molzahn_cases/nmwc3cyclic.m";
    # case_dir = "Molzahn_cases/nmwc4.m";
    # case_dir = "Molzahn_cases/nmwc5.m";
    # case_dir = "Molzahn_cases/nmwc14.m";
    # case_dir = "Molzahn_cases/nmwc24.m";
    # case_dir = "Molzahn_cases/nmwc57.m";
    # case_dir = "Molzahn_cases/nmwc118.m"; # seems to fail generating endpoints

    # SP.enable_parallel();
    t = run_case(case_dir; output_file="plot.pdf", rng);
    GC.gc(true);
    println(t)
end