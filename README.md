### Syntax

    run_case(case_dir::String; output_file::String="plot.pdf", rng::MersenneTwister=MersenneTwister());
    
This function compues the shortest path between two feasible points of the [MATPOWER](https://github.com/MATPOWER/matpower) OPF test case with file address specified by the String `case_dir`, and generates a PDF plot of the result. Keyword argument `output_file` is a String that specifies the output PDF filename (defaults to `"plot.pdf"`). Keyword argument `rng` (deprecated, not in use) specifies a Mersenne Twister RNG object, in case seed control is needed for replicability (defaults to `MersenneTwister()` a standard, uncontrolled instance of type `MersenneTwister`). The endpoints used for the shortest path algorithm are the solution of the minimum loss problem and the solution of the OPF problem, both for the test case.

### Examples
If the dependencies are not installed we can use the sample project provided in `Project.toml` (it is not clean though):

    ]activate .
    instantiate
    
After instantiating the sample project, or if all dependencies are already installed, we can run the code directly. For example:

    include("barrier_case_pdf.jl");
    run_case("MATPOWER/case9mod.m"; output_file="plot.pdf");
   
