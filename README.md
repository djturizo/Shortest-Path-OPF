### Syntax

    run_case(case_dir::String;
    u_start=nothing, u_end=nothing,
    μ=1e-5, μ_large=0.0125, K=9, rng::MersenneTwister=MersenneTwister(),
    print_str=false, print_K=false, disable_msgs=false);
    
This function compues the shortest path between two feasible points of the [MATPOWER](https://github.com/MATPOWER/matpower) OPF test case with file address specified by the String `case_dir`, and outputs the tuple `(max_violation, max_violation_after, path_diff_pc, fobj_diff_pc, total_time)` with values corresponding to the metrics used in out [paper](https://arxiv.org/abs/2408.02172). Keyword arguments `u_start` and `u_end` are the starting and ending points of the desired path. If the endpoints are not provided, `u_start` is taken as the solution of the minimum losss problem, and `u_end` is taken as the solution of the ACOPF problem. Keyword arguments `μ` and `μ_large` are the barrier parameters for the shorest path step and feasible path step of the algorithm, respectively; they should not be changed unless you know what you are doing. Keyword argument `rng` (deprecated, not in use) specifies a Mersenne Twister RNG object, in case seed control is needed for replicability (defaults to `MersenneTwister()` a standard, uncontrolled instance of type `MersenneTwister`). Keyword argument `disable_msgs` disables warning messages if set to `true`. Keyword arguments `print_str` and `print_K` are not for public use and should be left as `false`.

### Examples
If the dependencies are not installed we can use the sample project provided in `Project.toml` (it is not clean though):

    ]activate .
    instantiate
    
After instantiating the sample project, or if all dependencies are already installed, we can run the code directly. For example:

    include("barrier_case_time_v2.jl");
    run_case("MATPOWER/case9mod.m");
   
