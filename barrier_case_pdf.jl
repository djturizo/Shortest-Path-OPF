using LinearAlgebra, Random, Suppressor, StatProfilerHTML, Profile

import PowerModels as PM, Ipopt as Ipopt
import Plots as plt, ColorSchemes as CS

include("power_flow_rect_qc.jl")
import .PowerFlowRectQC as QC
include("shortest_path.jl")
import .ShortestPathOPF as SP

using Infiltrator
Infiltrator.clear_disabled!()

AUTO_RUN = false; # Flag to run a test case automatically on loading


function run_case(case_dir::String; output_file::String="plot.pdf",
    rng::MersenneTwister=MersenneTwister())
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
    μ = 1e-6
    nu = 1e6


    # Load test case
    mpc = QC.load_case(case_dir)
    case_qc = QC.compute_qc(mpc, !true)
    case_data = SP.load_case(case_dir, case_qc)
    qc_data = case_data.qc_data
    n = qc_data.n
    ind_u = case_data.ind_u
    dim_u = length(ind_u)
    u0 = qc_data.u0
    PdQd = case_data.PdQd

    # Get start point as the solution of the minimum loss problem,
    # and end point as solution of the OPF.
    u_start_full, x_start, u_end_full, x_end = SP.get_endpoints(
        mpc, case_data, tol_inner, rng)
    # Correct endpoints so as to not include loads
    u_start = u_start_full[ind_u] + PdQd[ind_u]
    u_end = u_end_full[ind_u] + PdQd[ind_u]
    # New base case is the start point
    u0 = deepcopy(u_start_full)
    println("end-point distance (full): $(max(abs.(u_start_full-u_end_full)...))")
    u_diff = [u_start_full[1:end] u_end_full[1:end]][
        setdiff(1:end,ind_u),:]
    println("end-point distance (fixed entries): $(max(abs.(u_diff[:,1] - u_diff[:,2])...))")
    println()
    GC.gc(true) # force GC to clean any Ipopt residues

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

    # Main animation function
    const_frames = Vector{Vector{Float64}}()
    function add_frames(tvec, const_frames)
        plt.gr(size=(640,480), legend=:none)
        p1 = plt.plot(tvec[[1,end]], [0.0, 0.0], linewidth=2.0, c=:red)
        nk = length(const_frames)
        for k = 1:nk
            plt.plot!(tvec, const_frames[k], linewidth=1.0*(k/nk)+1.0,
                c=get(CS.devon, 1.0-k/nk))
        end
        plt.xlims!((tvec[1], tvec[end]))
        plt.display(p1)
    end

    # Profile.clear()
    # Profile.init(n=10^8, delay=0.01)
    # @profile begin
    # try
    #--------------------------------------------------------------------
    # Find initial feasible path
    μ_large = 1e-1
    get_pk = (v) -> map((k) -> v[(k*dim_u+1):((k+1)*dim_u)],
        0:(n_points-1)) # does NOT exclude extreme points
    get_cons_pk = (v) -> map((cons) -> max(cons...), path_oracle(get_pk(v))[1])
    v0, x_pk0, beta_vec, v0_hist, _ = SP.get_feasible_path(case_data, u_start,
        u_end, tvec, tol_outer, tol_inner, tol_pf, iter_max_inner,
        iter_max_pf, μ_large, nu, u0, true)
    n_v0 = length(v0_hist)
    for k = 1:n_v0
        const_frames = vcat(const_frames, get_cons_pk.(v0_hist[k]))
    end

    if !(max(beta_vec...) > tol_inner)
        # Solve shortest path problem
        v0, x_pk0, exit_flag, v_hist = SP.get_shortest_path(case_functions,
            tvec, v0; x_pk0, tol_outer, alpha_min=tol_inner,
            iter_max=iter_max_inner, μ, nu, save_hist=true)
        const_frames = vcat(const_frames, get_cons_pk.(v_hist))
    end
    #--------------------------------------------------------------------
    # catch; end
    # end
    # statprofilehtml(path="time_test")
    # Profile.clear()
    # return

    add_frames(tvec, const_frames)
    plt.savefig(output_file)
    SP.close_parallel()
    return
end

# Run automatically
if AUTO_RUN
    # Set up rng with fixed seed for reproducibility
    # This seed gives "disconected" endpoints for case9mod
    # This seed gives a non-trival path for case39_epri
    # This seed gives "disconected" endpoints for case60_c
    # This seed gives a non-trival path for case73_ieee
    rng = MersenneTwister(UInt32[0x2070c972, 0x521b92e3, 0xaf734195, 0x223eab32]);
    # Load test case
    case_dir = "MATPOWER/case9mod.m";
    # case_dir = "MATPOWER/pglib_opf_case14_ieee.m";
    # case_dir = "MATPOWER/pglib_opf_case24_ieee_rts.m"; # Good one
    # case_dir = "MATPOWER/pglib_opf_case30_ieee.m";
    # case_dir = "MATPOWER/pglib_opf_case39_epri.m"; # Good one
    # case_dir = "MATPOWER/pglib_opf_case57_ieee.m"; # Good one
    # case_dir = "MATPOWER/pglib_opf_case60_c.m"; # Good one
    # case_dir = "MATPOWER/pglib_opf_case73_ieee_rts.m"; # Good one
    # case_dir = "MATPOWER/pglib_opf_case118_ieee.m"; # Good one
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
    # case_dir = "Molzahn_cases/nmwc118.m";

    # SP.enable_parallel();
    run_case(case_dir; output_file="plot.pdf", rng);
    GC.gc(true)
end