### Syntax

    run_case(case_dir::String; output_file::String, br_fac::Float64, rng::MersenneTwister);
    
This function runs the shortest path algorithm using the [MATPOWER](https://github.com/MATPOWER/matpower) test case with file address specifi4ed by the String `case_dir`, and generates a PDF plot of the result. Keyword argument `output_file` is a String that specifies the output PDF filename (defaults to `"plot.pdf"`). Keyword argument `br_fac` specifies the branch limit reduction factor for current inequalities (defaults to `0.8`), it is best left unchanged. Keyword argument `rng` specifies a Mersenne Twister RNG object, in case seed control is needed for replicability (defaults to `MersenneTwister()` a standard, uncontrolled instance of type `MersenneTwister`).

### Examples
If the dependencies are not installed we can use the sample project provided in `Project.toml` (it is not clean though):

    ]activate .
    instantiate
    
After instantiating the sample project, or if all dependencies are already installed, we can run the code directly. For example:

    include("barrier_case_pdf.jl");
    run_case("MATPOWER/case9mod.m"; output_file="plot.pdf");
    
If we want replicability of the results we may fix the rng seed as follows:

    rng = MersenneTwister(UInt32[0x2070c972, 0x521b92e3, 0xaf734195, 0x223eab32]);
    include("barrier_case_pdf.jl");
    run_case("MATPOWER/case9mod.m"; output_file="plot.pdf", rng=rng);
    
You may use any other seed apart from `[0x2070c972, 0x521b92e3, 0xaf734195, 0x223eab32]`, as long as it is a 4-element vector of `UInt32` values.
