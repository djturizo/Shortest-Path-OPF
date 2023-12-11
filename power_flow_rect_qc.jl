module PowerFlowRectQC
# We assume that there is only one slack bus

using LinearAlgebra
import SparseArrays as SA
import PowerModels as PM, ForwardDiff as FD,
    SparseDiffTools as SDT, SparsityDetection as SD,
    MathOptSymbolicAD as SAD


const θ_limit = 1.5707 # must be close, but less than pi/2
const θ_ignore = 1.5706 # must be close, but less than 'θ_limit'


macro fullprint(var)
    return esc(:(show(stdout, "text/plain", $var); println()))
end


function parse_case(mpc)
    # Compute admittance matrix
    n = length(mpc["bus"])
    Y = PM.calc_basic_admittance_matrix(mpc)
    G = real.(Y)
    B = imag.(Y)
    Yrec = SA.sparse_vcat(SA.sparse_hcat(G, -B),
        SA.sparse_hcat(B, G))

    # Voltage phasors
    result = PM.compute_ac_pf(mpc)
    if result["termination_status"]
        # Power flow converged
        V0 = zeros(ComplexF64, n)
        for (key, bus) in result["solution"]["bus"]
            V0[parse(Int64, key)] = bus["vm"] *
                exp(im*bus["va"])
        end
    else
        println("Power flow did not converge! " *
            "Using flat start as the operating point")
        V0 = ones(ComplexF64, n)
    end
    x0 = vcat(real.(V0), imag.(V0))

    # Generator variables
    Vg = zeros(Float64, n)
    Pg = zeros(Float64, n)
    Pg_max = zeros(Float64, n)
    Pg_min = zeros(Float64, n)
    Qg_max = zeros(Float64, n)
    Qg_min = zeros(Float64, n)
    # Generator costs (only order <=2 terms considered)
    c2_gen = zeros(Float64, n)
    c1_gen = zeros(Float64, n)
    c0_gen = zeros(Float64, n)
    # Power demands
    Pd = zeros(Float64, n)
    Qd = zeros(Float64, n)

    # Logical indices of PQ, PV and slack buses
    c_pv = fill(false, n)
    c_vd = fill(false, n)
    c_pq = fill(false, n)
    c_pvq = fill(false, n) # PV and PQ nodes
    c_pvd = fill(false, n) # PV and slack nodes

    # Parse inputs from "bus" dict
    for (_, bus) in mpc["bus"]
        key = bus["bus_i"]
        if bus["bus_type"] == 3
            # Slack bus
            c_vd[key] = true
            c_pvd[key] = true
        else
            c_pvq[key] = true
            if bus["bus_type"] == 2
                # PV bus
                c_pv[key] = true
                c_pvd[key] = true
            else
                # PQ bus
                c_pq[key] = true
            end
        end
    end

    # Throw an error if there is more than 1 slack node
    n_vd = sum(c_vd)
    if n_vd != 1
        error("The power system must have exactly one slack node, " *
            "but this system has " * string(n_vd) * "slack nodes.")
    end

    # Parse inputs from "gen" dict
    # Multiple generators may be connected to the same bus,
    # the power parameters are added to represent a single
    # machine per bus.
    bus_gen_count = zeros(Int64, n)
    multiple_gen_flag = false
    for (_, gen) in mpc["gen"]
        bus_key = gen["gen_bus"]
        Pg[bus_key] += gen["pg"]
        Pg_max[bus_key] += gen["pmax"]
        Pg_min[bus_key] += gen["pmin"]
        Qg_max[bus_key] += gen["qmax"]
        Qg_min[bus_key] += gen["qmin"]
        c0_gen[bus_key] += gen["cost"][end]
        c1_gen[bus_key] += gen["cost"][end-1]
        c2_gen[bus_key] += gen["cost"][end-2]
        if bus_gen_count[bus_key] > 0 && !multiple_gen_flag
            @warn "The system has multiple generators in a single bus, " *
                "procceding to merge them. All information besides " *
                "basic OPF parameters will be lost."
            multiple_gen_flag = true
        else
            # The voltage for this bus will be the one of the first
            # parsed gen.
            Vg[bus_key] = gen["vg"]
        end
        bus_gen_count[bus_key] += 1
    end
    # We assume that production among generator in the same bus is
    # split equally, so we adjust the cost accordingly. This assumption can be
    # inconsistent with adding P and Q limits. The more appropriate assumption
    # would be to prorrate the cost coefficients of each generator according to
    # their own P limits, but we are trying to keep things simple for now.
    # TODO: Prorrate cost coefficients properly.
    c1_gen[bus_gen_count.>0] ./= bus_gen_count[bus_gen_count.>0]
    c2_gen[bus_gen_count.>0] ./= bus_gen_count[bus_gen_count.>0].^2

    # Parse inputs from "load" dict
    # Just in case, we also add load powers
    for (_, load) in mpc["load"]
        key = load["load_bus"]
        Pd[key] += load["pd"]
        Qd[key] += load["qd"]
    end

    # Compute OPF quadratic constraints and objective
    # TODO: efficient sparse matrix assignment
    # Vmin and Vmax constraints
    opf_Vmax_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, n)
    opf_Vmax_J = SA.spzeros(Float64,Int64, n,2*n)
    opf_Vmax_c = zeros(Float64, n)
    opf_Vmin_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, n)
    opf_Vmin_J = SA.spzeros(Float64,Int64, n,2*n)
    opf_Vmin_c = zeros(Float64, n)
    for (_, bus) in mpc["bus"]
        key = bus["bus_i"]
        if bus["bus_type"] == 3
            # Slack bus, V constraints can be in u or in x
            # We choose to write the constraints in x
            opf_Vmax_H[key] = SA.sparse([key, n+key], [key, n+key],
                [2.0, 2.0], 2*n, 2*n)
            opf_Vmax_c[key] += -bus["vmax"]^2
            opf_Vmin_c[key] += bus["vmin"]^2
        elseif bus["bus_type"] == 2
            # PV bus, V constraints are in u
            opf_Vmax_H[key] = SA.spzeros(Float64,Int64, 2*n,2*n)
            opf_Vmax_J[key,:] = SA.sparse([1], [n+key],
                [1.0], 1, 2*n)
            opf_Vmax_c[key] += -bus["vmax"]^2
            opf_Vmin_c[key] += bus["vmin"]^2
        else
            # PQ bus, V constraints are in x
            opf_Vmax_H[key] = 2.0.*SA.sparse([key, n+key], [key, n+key],
                [1.0, 1.0], 2*n, 2*n)
            opf_Vmax_c[key] += -bus["vmax"]^2
            opf_Vmin_c[key] += bus["vmin"]^2
        end
        opf_Vmin_H[key] = -opf_Vmax_H[key]
        opf_Vmin_J[key,:] = -opf_Vmax_J[key,:]
    end

    # Generator P and Q constraints
    # Constraints are written in terms of net injection, Pg-Pd and Qg-Qd
    # Consider only one generator per node, the equivalent one.
    ind_pvd = findall(c_pvd)
    g = length(ind_pvd)
    opf_Pmax_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, g)
    opf_Pmax_J = SA.spzeros(Float64,Int64, g,2*n)
    opf_Pmax_c = zeros(Float64, g)
    opf_Pmin_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, g)
    opf_Pmin_J = SA.spzeros(Float64,Int64, g,2*n)
    opf_Pmin_c = zeros(Float64, g)
    opf_Qmax_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, g)
    opf_Qmax_J = SA.spzeros(Float64,Int64, g,2*n)
    opf_Qmax_c = zeros(Float64, g)
    opf_Qmin_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, g)
    opf_Qmin_J = SA.spzeros(Float64,Int64, g,2*n)
    opf_Qmin_c = zeros(Float64, g)
    c = 0
    i_slack = 0 # index of slack generator
    for key in ind_pvd
        if c_vd[key]
            # Slack generator, P constraints are in x
            # Put slack constraints as last entries in the array
            # Jacobian at origin is zero in this case
            # TODO: Add compatibility for multiple slack nodes
            if i_slack != 0
                error("Systems with multiple slack generators are not supported.")
            end
            i_slack = c + 1
            c = g
            opf_Pmax_H[c] = SA.sparse([key], [1], [1.0],
                2*n, 1) * Yrec[key:key,:]
            opf_Pmax_H[c] += SA.sparse([n+key], [1], [1.0],
                2*n, 1) * Yrec[(n+key):(n+key),:]
            opf_Pmax_H[c] += transpose(opf_Pmax_H[c])
            # opf_Pmax_J[c,:] = SA.sparse([1], [key],
            #     [1.0], 1, 2*n)
        else
            c += 1
            # P constraints are in u
            opf_Pmax_H[c] = SA.spzeros(Float64,Int64, 2*n,2*n)
            opf_Pmax_J[c,:] = SA.sparse([1], [key],
                [1.0], 1, 2*n)
        end
        opf_Pmax_c[c] = -(Pg_max[key] - Pd[key])
        opf_Pmin_H[c] = -opf_Pmax_H[c]
        opf_Pmin_J[c,:] = -opf_Pmax_J[c,:]
        opf_Pmin_c[c] = Pg_min[key] - Pd[key]
        # Q constraints are in x
        opf_Qmax_H[c] = -SA.sparse([key], [1], [1.0],
            2*n, 1) * Yrec[(n+key):(n+key),:]
        opf_Qmax_H[c] += SA.sparse([n+key], [1], [1.0],
            2*n, 1) * Yrec[key:key,:]
        opf_Qmax_H[c] += transpose(opf_Qmax_H[c])
        opf_Qmax_c[c] = -(Qg_max[key] - Qd[key])
        opf_Qmin_H[c] = -opf_Qmax_H[c]
        opf_Qmin_c[c] = Qg_min[key] - Qd[key]
        if c_vd[key]
            # Return counter to its original state
            c = i_slack - 1
        end
    end
    if i_slack == 0
        error("The system does not have a slack generator.")
    end

    # Objective function (generator cost), PV units first
    opf_fobj_Hu = SA.spdiagm(2*n, 2*n, 0 => 2 .* c2_gen)
    opf_fobj_Ju = vcat(c1_gen, zeros(Float64, n))'
    opf_fobj_c = sum(c0_gen)
    # Remove slack generator terms from coefficients (we assume only 1 slack unit,
    # the first one)
    # TODO: enforce consistency of this assumption outside of this function.
    ind_vd = findall(c_vd)[1]
    opf_fobj_Hu[ind_vd,ind_vd] = 0
    opf_fobj_Ju[ind_vd] = 0
    opf_fobj_c -= c0_gen[ind_vd]
    # Objective function written in terms of net injection, Pg-Pd and Qg-Qd,
    # so the constant and linear terms must be adjusted accordingly.
    # Consider only one generator per node, the equivalent one.
    PdQd = vcat(Pd, Qd)
    opf_fobj_c += opf_fobj_Ju * PdQd
    opf_fobj_Ju += (PdQd') * opf_fobj_Hu
    opf_fobj_c += 0.5 * ((PdQd') * opf_fobj_Hu * PdQd)
    # Slack generator cost has to be written in term of the vector of voltages, x
    cost_vd = [c2_gen[ind_vd]; c1_gen[ind_vd]; c0_gen[ind_vd]]
    # P_vd = Pd_vd + 0.5 * x'*H_vd*x
    Pd_vd = Pd[ind_vd]
    H_vd = deepcopy(opf_Pmax_H[g]) # last constraint is always the slack unit


    # Branch angle and I^2 constraints (constraints are in x)
    b = length(mpc["branch"])
    opf_I2max_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, 2*b)
    opf_I2max_J = SA.spzeros(Float64,Int64, 2*b, 2*n)
    opf_I2max_c = zeros(Float64, 2*b)
    opf_I2max_buses = zeros(Int64, 2*b)
    c_I2 = 0
    opf_θmax_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, b)
    opf_θmax_J = SA.spzeros(Float64,Int64, b, 2*n)
    opf_θmax_c = zeros(Float64, b)
    c_θmax = 0
    opf_θmin_H = Array{SA.SparseMatrixCSC{Float64,Int64}}(undef, b)
    opf_θmin_J = SA.spzeros(Float64,Int64, b, 2*n)
    opf_θmin_c = zeros(Float64, b)
    c_θmin = 0
    for (_, branch) in mpc["branch"]
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]

        # REMEMBER: this is a constraint in I SQUARED!!!
        I2max = branch["rate_a"]^2
        if I2max > 0.0
            ys = 1/(branch["br_r"] + im*branch["br_x"])
            a = branch["tap"]*exp(im*branch["shift"])
            jb_fr = im*branch["b_fr"]
            jb_to = im*branch["b_to"]
            # I^2 contraint at the "from" node
            c_I2 += 1
            Y_fr = SA.sparse([f_bus, f_bus], [f_bus, t_bus],
                [(ys + jb_fr)/abs(a)^2, -ys/conj(a)], n, n)
            Yrec_fr = vcat(hcat(real.(Y_fr), -imag.(Y_fr)),
                hcat(imag.(Y_fr), real.(Y_fr)))
            opf_I2max_H[c_I2] = 2.0 .* (transpose(Yrec_fr)*Yrec_fr)
            opf_I2max_c[c_I2] = -I2max
            opf_I2max_buses[c_I2] = f_bus
            # I^2 contraint at the "from" node
            c_I2 += 1
            Y_to = SA.sparse([t_bus, t_bus], [f_bus, t_bus],
                [-ys/a, ys + jb_to], n, n)
            Yrec_to = vcat(hcat(real.(Y_to), -imag.(Y_to)),
                hcat(imag.(Y_to), real.(Y_to)))
            opf_I2max_H[c_I2] = 2.0 .* (transpose(Yrec_to)*Yrec_to)
            opf_I2max_c[c_I2] = -I2max
            opf_I2max_buses[c_I2] = t_bus
        end

        # Make max. angle constraint, if abs. limit is less than ~90 deg.
        # Angle equation (from bus 1 to bus 2)
        # Im((V₁ᵣ + jV₁ᵢ)(V₂ᵣ - jV₂ᵢ)) / Re((V₁ᵣ + jV₁ᵢ)(V₂ᵣ - jV₂ᵢ)) ≤ tan(θmax)
        # (V₁ᵢV₂ᵣ - V₁ᵣV₂ᵢ) / (V₁ᵣV₂ᵣ + V₁ᵢV₂ᵢ) ≤ tan(θmax)
        # 0.5(2V₁ᵢV₂ᵣ - 2V₁ᵣV₂ᵢ  - 2tan(θmax)V₁ᵣV₂ᵣ - 2tan(θmax)V₁ᵢV₂ᵢ) ≤ 0
        θmax = branch["angmax"]
        if abs(θmax) < θ_ignore
            c_θmax += 1
            tanmax = tan(θmax)
            opf_θmax_H[c_θmax] = SA.sparse([f_bus, f_bus, t_bus, n+f_bus],
                [t_bus, n+t_bus, n+f_bus, n+t_bus],
                [-tanmax, -1.0, 1.0, -tanmax], 2*n, 2*n)
            opf_θmax_H[c_θmax] += transpose(opf_θmax_H[c_θmax])
        end

        # Make min. angle constraint, if abs. limit is more than ~90 deg.
        # Angle equation (from bus 1 to bus 2)
        # Im((V₁ᵣ + jV₁ᵢ)(V₂ᵣ - jV₂ᵢ)) / Re((V₁ᵣ + jV₁ᵢ)(V₂ᵣ - jV₂ᵢ)) ≥ tan(θmin)
        # (V₁ᵢV₂ᵣ - V₁ᵣV₂ᵢ) / (V₁ᵣV₂ᵣ + V₁ᵢV₂ᵢ) ≥ tan(θmin)
        # 0.5(-2V₁ᵢV₂ᵣ + 2V₁ᵣV₂ᵢ  + 2tan(θmax)V₁ᵣV₂ᵣ + 2tan(θmax)V₁ᵢV₂ᵢ) ≤ 0
        θmin = branch["angmin"]
        if abs(θmin) < θ_ignore
            c_θmin += 1
            tanmin = tan(θmin)
            opf_θmin_H[c_θmin] = SA.sparse([f_bus, f_bus, t_bus, n+f_bus],
                [t_bus, n+t_bus, n+f_bus, n+t_bus],
                -[-tanmin, -1.0, 1.0, -tanmin], 2*n, 2*n)
            opf_θmin_H[c_θmin] += transpose(opf_θmin_H[c_θmin])
        end
    end
    # Keep only contraints that were built
    opf_I2max_H = opf_I2max_H[1:c_I2]
    opf_I2max_J = opf_I2max_J[1:c_I2,:]
    opf_I2max_c = opf_I2max_c[1:c_I2]
    opf_I2max_buses = opf_I2max_buses[1:c_I2]
    opf_θmax_H = opf_θmax_H[1:c_θmax]
    opf_θmax_J = opf_θmax_J[1:c_θmax,:]
    opf_θmax_c = opf_θmax_c[1:c_θmax]
    opf_θmin_H = opf_θmin_H[1:c_θmin]
    opf_θmin_J = opf_θmin_J[1:c_θmin,:]
    opf_θmin_c = opf_θmin_c[1:c_θmin]

    # Concatenate constraints in u
    # TODO: efficient sparse matrix concatenation
    opf_u_H = vcat(opf_Vmax_H[c_pv], opf_Vmin_H[c_pv],
        opf_Pmax_H[1:(end-1)], opf_Pmin_H[1:(end-1)])
    opf_u_J = vcat(opf_Vmax_J[c_pv,:], opf_Vmin_J[c_pv,:],
        opf_Pmax_J[1:(end-1),:], opf_Pmin_J[1:(end-1),:])
    opf_u_c = vcat(opf_Vmax_c[c_pv,:], opf_Vmin_c[c_pv,:],
        opf_Pmax_c[1:(end-1)], opf_Pmin_c[1:(end-1)])
    # @fullprint [length(opf_Vmax_c[c_pv,:]);
    #     length(opf_Vmin_c[c_pv,:]); length(opf_Pmax_c[1:(end-1)]);
    #     length(opf_Pmin_c[1:(end-1)])]

    # Concatenate constraints in x
    # I² constraints are placed first, so they are easier to locate.
    opf_x_H = vcat(opf_I2max_H, opf_Vmax_H[.!c_pv], opf_Vmin_H[.!c_pv],
        opf_Qmax_H, opf_Qmin_H, [opf_Pmax_H[end]], [opf_Pmin_H[end]],
        opf_θmax_H, opf_θmin_H)
    opf_x_J = vcat(opf_I2max_J, opf_Vmax_J[.!c_pv,:], opf_Vmin_J[.!c_pv,:],
        opf_Qmax_J, opf_Qmin_J, opf_Pmax_J[end:end,:], opf_Pmin_J[end:end,:],
        opf_θmax_J, opf_θmin_J)
    opf_x_c = vcat(opf_I2max_c, opf_Vmax_c[.!c_pv,:], opf_Vmin_c[.!c_pv,:],
        opf_Qmax_c, opf_Qmin_c, opf_Pmax_c[end], opf_Pmin_c[end],
        opf_θmax_c, opf_θmin_c)
    # @fullprint [length(opf_Vmax_c[.!c_pv,:]);
    #     length(opf_Vmin_c[.!c_pv,:]); length(opf_Qmax_c);
    #     length(opf_Qmin_c); length(opf_Pmax_c[end]); length(opf_Pmin_c[end]);
    #     length(opf_I2max_c)]


    # Declare type of output variables and return them
    case = @NamedTuple{n::Int64, g::Int64,
        Yrec::SA.SparseMatrixCSC{Float64,Int64},
        Y::SA.SparseMatrixCSC{ComplexF64,Int64},
        V0::Vector{ComplexF64}, x0::Vector{Float64},
        mpc::Dict{String, Any}}(
        (n, g, Yrec, Y, V0, x0, mpc))
    choosers = @NamedTuple{vd::Vector{Bool}, pv::Vector{Bool},
        pq::Vector{Bool}, pvq::Vector{Bool}, pvd::Vector{Bool}}(
        (c_vd, c_pv, c_pq, c_pvq, c_pvd))
    input = @NamedTuple{Vg::Vector{Float64}, Pg::Vector{Float64},
        Pd::Vector{Float64}, Qd::Vector{Float64}}(
        (Vg[c_pvd], Pg[c_pvd], Pd, Qd))
    opf_u = @NamedTuple{c::Array{Float64},
        J::SA.SparseMatrixCSC{Float64,Int64},
        H::Array{SA.SparseMatrixCSC{Float64,Int64}}}(
        (opf_u_c, opf_u_J, opf_u_H))
    opf_x = @NamedTuple{c::Array{Float64},
        J::SA.SparseMatrixCSC{Float64,Int64},
        H::Array{SA.SparseMatrixCSC{Float64,Int64}},
        rate_con_buses::Array{Int64}}(
        (opf_x_c, opf_x_J, opf_x_H, opf_I2max_buses))
    opf_fobj = @NamedTuple{cu::Float64,
        Ju::SA.SparseMatrixCSC{Float64,Int64},
        Hu::SA.SparseMatrixCSC{Float64,Int64},
        cost_vd::Vector{Float64},
        Pd_vd::Float64,
        H_vd::SA.SparseMatrixCSC{Float64,Int64}}(
        (opf_fobj_c, opf_fobj_Ju, opf_fobj_Hu, cost_vd, Pd_vd, H_vd))
    return case, choosers, input, opf_u, opf_x, opf_fobj, mpc
end


function load_case(case_input::String)
    mpc = PM.parse_file(case_input; validate=false)
    PM.make_per_unit!(mpc)
    PM.correct_voltage_angle_differences!(mpc, θ_limit)
    PM.correct_network_data!(mpc)
    return PM.make_basic_network(mpc)
end


function parse_case(case_input::String)
    # Load test case
    mpc = load_case(case_input)
    return parse_case(mpc)
end


function pf_hessian(case, ind)::Tuple{
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    SA.SparseMatrixCSC{Float64,Int64}}

    n = case.n
    Yrec = case.Yrec
    # Power flow Hessian
    Hpf = fill(SA.spzeros(Float64, 2*n, 2*n), 2*n)
    # Active power Hessians
    HP = fill(SA.spzeros(Float64, 2*n, 2*n), n)
    # Reactive power Hessians
    HQ = fill(SA.spzeros(Float64, 2*n, 2*n), n)
    # Squared voltage Hessians
    HV = fill(SA.spzeros(Float64, 2*n, 2*n), n)
    # Total loss Hessian
    Hloss = SA.spzeros(Float64, 2*n, 2*n)
    for k in 1:n
        # Compute P Hessians
        HP[k] = SA.sparse([k], [1], [1.0], 2*n, 1) * Yrec[k:k,:]
        HP[k] += SA.sparse([n+k], [1], [1.0], 2*n, 1) *
            Yrec[(n+k):(n+k),:]
        HP[k] += transpose(HP[k])
        Hloss += HP[k] # Loss Hessian is the sum of all P Hessians
        # Compute Q Hessians
        HQ[k] = -SA.sparse([k], [1], [1.0], 2*n, 1) *
            Yrec[(n+k):(n+k),:]
        HQ[k] += SA.sparse([n+k], [1], [1.0], 2*n, 1) *
            Yrec[k:k,:]
        HQ[k] += transpose(HQ[k])
        # Compute V² Hessians
        HV[k] = SA.sparse([k, n+k], [k, n+k],
            [2.0, 2.0], 2*n, 2*n)
    end
    # Slack bus Hessian is 0, we don't have to compute it
    Hpf[ind.pvq] = HP[ind.pvq] # P equations
    Hpf[.+(n,ind.pq)] = HQ[ind.pq] # Q equations
    Hpf[.+(n,ind.pv)] = HV[ind.pv] # V² equations
    return Hpf, HP, HQ, HV, Hloss
end


function pf_jacobian(x::Vector{Float64}, ind_vd::Int64,
    Hpf::Vector{SA.SparseMatrixCSC{Float64,Int64}}
    )::SA.SparseMatrixCSC{Float64,Int64}
    # Compute the power flow Jacobian from the Hessians
    # For simplicity, we compute the transposed Jacobian
    n = div(length(x), 2)
    x_sp = SA.sparse(x)
    nzval = Float64[]
    rowval = Int64[]
    colptr = ones(Int64, 2*n+1)
    # Build variables for direct constructor call
    for i in 1:(2*n)
        if i<=n && i == ind_vd
            # Slack bus, Hessian is zero
            # but Jacobian column is not
            col_i = SA.sparsevec([n+i], [1.0], 2*n)
        elseif (i-n) == ind_vd
            # Slack bus, Hessian is zero
            # but Jacobian column is not
            col_i = SA.sparsevec([i-n], [1.0], 2*n)
        else
            # For any other bus, just get the
            # Jacobian column from the Hessian
            col_i = Hpf[i]*x_sp
        end
        nzval = vcat(nzval, col_i.nzval)
        rowval = vcat(rowval, col_i.nzind)
        colptr[i+1] = colptr[i] + length(col_i.nzval)
    end
    # Build transposed Jacobian
    Jt = SA.SparseMatrixCSC{Float64,Int64}(2*n, 2*n,
        colptr, rowval, nzval)
    return transpose(Jt)
end

# Power flow solver function
function solve_pf(u::Vector{Float64},
    Hpf::Vector{SA.SparseMatrixCSC{Float64,Int64}},
    ind_vd::Int64, n::Int64, tol::Float64,
    iter_max::Int64)::Tuple{Vector{Float64},Int64}
    # Flat start
    x = vcat(ones(Float64, n), zeros(Float64, n))
    fx = zeros(Float64, 2*n)
    err = tol + 1.0
    # Newton iteration
    for iter in 0:iter_max
        # Compute power flow
        for i in 1:(2*n)
            if i<=n && i == ind_vd
                # Slack bus, imag part equation
                fx[i] = x[n+i]
            elseif (i-n) == ind_vd
                # Slack bus, real part equation
                fx[i] = x[i-n]
            else
                fx[i] = 0.5*(transpose(x)*(Hpf[i]*x))
            end
            fx[i] -= u[i]
        end
        # Check convergence
        err = max(abs.(fx)...)
        if err <= tol || iter >= iter_max
            break
        end
        # Compute power flow Jacobian.
        Jx = pf_jacobian(x, ind_vd, Hpf)
        # Do Newton step
        x -= Jx \ fx
    end
    # Report convergence
    exit_flag = 0::Int64
    if err <= tol
        exit_flag = 1::Int64
    end
    return x, exit_flag
end


function jac_matrices(n::Int64,
    ind_pv::Vector{Int64}, ind_vd::Int64)
    D_pv = SA.sparse(n.+ind_pv, n.+ind_pv, fill(1.0, length(ind_pv)),
        2*n, 2*n)
    D_vd = SA.sparse([ind_vd, n+ind_vd], [ind_vd, n+ind_vd],
        [1.0, 1.0], 2*n, 2*n)
    M_vd = SA.sparse([ind_vd, n+ind_vd], [n+ind_vd, ind_vd],
        [1.0, 1.0], 2*n, 2*n)
    i_vec = n .+ vcat(ind_pv, ind_pv)
    j_vec = vcat(ind_pv, n .+ ind_pv)
    return D_pv, D_vd, M_vd, i_vec, j_vec
end

function build_jac(x, YxR, YxI, n, D_pv, D_vd, M_vd,
    i_vec, j_vec, Yrec_mod)
    J_pv = SA.sparse(i_vec, j_vec, 2 .* x[j_vec], 2*n, 2*n)
    vR = x[1:n]
    vI = x[(n+1):end]
    return (SA.I - D_pv - D_vd) * (SA.spdiagm(0 => vcat(vR, vR),
    -n => vI, n => -vI) * Yrec_mod + SA.spdiagm(0 => vcat(YxR, YxR),
    -n => YxI, n => -YxI)) + J_pv + M_vd
end


function rect_jacobian(x::Vector{Float64},
    ind_pv::Vector{Int64}, ind_vd::Int64,
    Yrec_mod::SA.SparseMatrixCSC{Float64,Int64}
    )::SA.SparseMatrixCSC{Float64,Int64}
    n = div(length(x), 2)
    D_pv, D_vd, M_vd, i_vec, j_vec = jac_matrices(n, ind_pv, ind_vd)
    Yx = Yrec_mod * x
    YxR = Yx[1:n]
    YxI = Yx[(n+1):end]
    return build_jac(x, YxR, YxI, n, D_pv, D_vd, M_vd,
        i_vec, j_vec, Yrec_mod)
end


# Rectangular power flow solver function
function rect_pf(u::Vector{Float64},
    ind_pq::Vector{Int64}, ind_pv::Vector{Int64}, ind_vd::Int64,
    Yrec_mod::SA.SparseMatrixCSC{Float64,Int64},
    tol::Float64, iter_max::Int64,
    x0::Union{Vector{Float64},Nothing}=nothing,
    eval::Bool=false)::Tuple{Vector{Float64},Int64}
    n = div(length(u), 2)
    if !eval
        # Not in evaluation mode, get Jacobian data
        D_pv, D_vd, M_vd, i_vec, j_vec = jac_matrices(n, ind_pv, ind_vd)
    end
    if typeof(x0) <: Nothing
        # Starting guess not provided, use flat start
        x = vcat(ones(Float64, n), zeros(Float64, n))
    else
        x = deepcopy(x0)
    end
    fx = zeros(Float64, 2*n)
    err = tol + 1.0
    # Newton iteration
    for iter in 0:iter_max
        # Compute power flow
        Yx = Yrec_mod * x
        YxR = Yx[1:n]
        YxI = Yx[(n+1):end]
        fx[1:n] = x[1:n] .* YxR .- x[(n+1):end] .* YxI
        fx[.+(n,ind_pq)] = x[.+(n,ind_pq)] .* YxR[ind_pq] .+
            x[ind_pq] .* YxI[ind_pq]
        # PV node voltage equations
        fx[.+(n,ind_pv)] = x[ind_pv].^2 .+ x[.+(n,ind_pv)].^2
        # Slack node equations
        fx[ind_vd] = x[n+ind_vd]
        fx[n+ind_vd] = x[ind_vd]
        fx .-= u
        # If we are in evaluation mode, just return fx
        if eval
            exit_flag = (-1)::Int64 # indicates eval mode
            return fx, exit_flag
        end
        # Check convergence
        err = norm(fx, Inf)
        if err <= tol || iter >= iter_max
            break
        end
        # Compute power flow Jacobian.
        Jx = build_jac(x, YxR, YxI, n, D_pv, D_vd, M_vd,
            i_vec, j_vec, Yrec_mod)
        # Do Newton step
        x .-= Jx \ fx
    end
    # Report convergence
    exit_flag = 0::Int64 # indicates max. iterations reached
    if err <= tol
        exit_flag = 1::Int64 # indicates convergence
    end
    return x, exit_flag
end


# Power flow equations
function pf(x::Vector{Float64}, case, ind, input)::Vector{Float64}
    n = case.n
    xr = x[1:n]
    xi = x[(n+1):(2*n)]
    ic = case.Yrec * x
    ir = ic[1:n]
    ii = ic[(n+1):(2*n)]
    dP = xr .* ir + xi .* ii + input.Pd
    dQ = xi .* ir - xr .* ii + input.Qd
    fx1 = zeros(Float64, n)
    fx2 = zeros(Float64, n)
    # Active power equations for PV and slack buses
    fx1[ind.pvd] = dP[ind.pvd] - input.Pg
    # Slack bus imag part equation
    fx1[ind.vd] = xi[ind.vd]
    # Active power equations for PQ nodes
    fx1[ind.pq] = dP[ind.pq]
    # Reactive power equations for PQ buses
    fx2[ind.pq] = dQ[ind.pq]
    # Squared voltage magnitude equation for slack and PV buses
    fx2[ind.pvd] = xr[ind.pvd] .^ 2 + xi[ind.pvd] .^ 2 -
        input.Vg .^ 2
    # Slack bus real part equation
    fx2[ind.vd] = xr[ind.vd] - input.Vg[ind.gen_vd]
    return vcat(fx1, fx2)
end


# i-th scalar power flow equation
function pf_i(x::AbstractVector{T}, i::Int64, case, ind,
    u)::T where {T<:Number}

    n = case.n;
    k = i % n
    k = (k == 0 ? n : k)::Int64
    xr = x[k]
    xi = x[n+k]
    ir = (case.Yrec[k:k,:] * x)[1]
    ii = (case.Yrec[(n+k):(n+k),:] * x)[1]
    if i <= n
        if k == ind.vd
            # Slack bus imag part equation
            f_i = xi
        else
            # Active power equations for PV or PQ bus
            f_i = xr .* ir + xi .* ii
        end
    else
        if k == ind.vd
            # Slack bus real part equation
            f_i = xr
        else
            if insorted(k, ind.pq)
                # Reactive power equations for PQ bus
                f_i = xi .* ir - xr .* ii
            else
                # Squared V² equation for PV bus
                f_i = xr^2 + xi^2
            end
        end
    end
    return f_i - u[i]
end


function compare_derivatives(case, ind, input, u)
    n = case.n

    # Compute Jacobian and Hessians by formula
    Hpf, = pf_hessian(case, ind)
    J0 = pf_jacobian(case.x0, ind.vd, Hpf)
    println("Hessian benchmark (formula):")
    @time pf_hessian(case, ind)
    println("Jacobian benchmark (formula):")
    @time pf_jacobian(case.x0, ind.vd, Hpf)
    println("(NonZeros, Entries) = " *
        "($(length(J0.nzval)), $(length(case.x0)^2))")
    println("(MaxColor, Size)    = " *
        "($(max(SDT.matrix_colors(J0)...))," *
        " $(length(case.x0)))")

    # Wrapper for the full power flow equations
    pfw = (x) -> map((i) -> pf_i(x, i, case, ind, u), 1:(2*n))

    # # Compute Jacobian and Hessians by automatic differentiation
    # J0_ad = FD.jacobian(pfw, case.x0)
    # Hf_i = (i) -> FD.hessian(
    #     (x) -> pf_i(x, i, case, ind, u), case.x0)
    # Hf_ad = map(Hf_i, 1:(2*n))
    # println("Hessian benchmark (ForwardDiff):")
    # @time map(Hf_i, 1:(2*n))
    # println("Jacobian benchmark (ForwardDiff):")
    # @time FD.jacobian(pfw, case.x0)
    # println("Jacobian benchmark (SparseDiffTools):")
    # SDT.forwarddiff_color_jacobian(pfw, case.x0,
    #     colorvec=SDT.matrix_colors(J0))
    # @time SDT.forwarddiff_color_jacobian(pfw, case.x0,
    # colorvec=SDT.matrix_colors(J0))

    # Compare to verify correctness
    print("PF equations max-error: ")
    @fullprint max(abs.(pfw(case.x0))...)
    print("PFv2 eqs. max-error:    ")
    @fullprint max(abs.(pf(case.x0, case, ind, input))...)
    # print("PF Jacobian max-error:  ")
    # @fullprint max(abs.(J0_ad - J0)...)
    # print("PF Hessian max-error:   ")
    # @fullprint max([max(abs.(Hf_ad[i] - Hpf[i])...)
    #     for i in 1:(2*n)]...)
end


function compute_qc(case_input,
    check_derivatives::Bool = false)
    # Parse MATPOWER case
    case, c_old, input, opf_u, opf_x, opf_fobj, mpc = parse_case(case_input)
    n = case.n

    # SymbolicAD benchamrk
    if check_derivatives
        if typeof(case) <: String
            pm = PM.instantiate_model(
                case_input, PM.ACPPowerModel, PM.build_opf)
        else
            # 'case_input' is an MPC instance. Hopefully the same sintax works?
            # Haven't tried though
            # TODO: Complete this case
            error()
        end
        SAD._nlp_block_data(pm.model; backend = SAD.DefaultBackend())
        println("Full benchmark (SymbolicAD):")
        @time SAD._nlp_block_data(pm.model;
            backend = SAD.DefaultBackend())
    end

    # Keep only 1 slack bus (Vd bus), make the rest PV
    ind_vd = findall(c_old.vd)[1]
    c_vd = fill(false, n)
    c_vd[ind_vd] = true
    c_pv = deepcopy(c_old.pvd)
    c_pv[ind_vd] = false
    c_pvq = collect(c_old.pq .| c_pv)
    ind_gen_vd = findfirst(findall(c_old.pvd) .== ind_vd)
    c = @NamedTuple{vd::Vector{Bool}, pv::Vector{Bool},
        pq::Vector{Bool}, pvq::Vector{Bool}, pvd::Vector{Bool}}(
        (c_vd, c_pv, c_old.pq, c_pvq, c_old.pvd))

    # Cartesian indices
    ind = @NamedTuple{vd::Int64, pv::Vector{Int64},
        pq::Vector{Int64}, pvq::Vector{Int64},
        pvd::Vector{Int64}, gen_vd::Int64}(
        (ind_vd, findall(c.pv), findall(c.pq),
        findall(c.pvq), findall(c.pvd), ind_gen_vd))

    # Compute active power of slack->PV buses
    c_vd_to_pv = deepcopy(c_old.vd)
    c_vd_to_pv[ind_vd] = false
    ind_vd_to_pv = findall(c_vd_to_pv)
    ind_gen_vd_to_pv = c_vd_to_pv[c_old.pvd]
    k = 0::Int64
    for i in ind_vd_to_pv
        k += 1
        input.Pg[ind_gen_vd_to_pv[k]] =
            0.5*(transpose(case.x0)*(Hp[i]*case.x0)) + Pd[i]
    end

    # Vector of control variables
    u0 = -vcat(input.Pd, input.Qd)
    u0[ind.pvd] += input.Pg
    # imag part of slack bus voltage
    u0[ind.vd] = 0.0
    u0[.+(n,ind.pvd)] = input.Vg .^ 2
    # real part of slack bus voltage
    u0[.+(n,ind.vd)] = input.Vg[ind.gen_vd]

    # Compare derivatives if required
    if check_derivatives
        compare_derivatives(case, ind, input, u0)
    end

    # Modified admittance matrices for complex PF
    Dpv = SA.sparse(ind.pv, ind.pv, fill(0.5, length(ind.pv)),
        n, n)
    Dvd = SA.sparse([ind.vd], [ind.vd], [1.0], n, n)
    zYv = Dpv * case.Y + 2im .* Dpv
    vYz = conj((SA.I - Dpv - Dvd) * case.Y)
    Yv = im .* Dvd

    # Modified admittance matrices for rectangular PF and Jacobian
    G = real.(case.Y)
    B = imag.(case.Y)
    Yrec_mod = SA.sparse_vcat(SA.sparse_hcat(G, -B),
        SA.sparse_hcat(-B, -G))

    # Compute PF Hessians and Jacobian
    Hpf, HP, HQ, HV, Hloss = pf_hessian(case,ind)
    J0 = rect_jacobian(case.x0, ind.pv, ind.vd, Yrec_mod)
    # Jacobian at origin for PF equations
    Jpf = rect_jacobian(zeros(Float64, 2*n), ind.pv, ind.vd, Yrec_mod)

    # Quadratic constraints data to be returned
    qc_data = @NamedTuple{n::Int64, x0::Vector{Float64},
        u0::Vector{Float64},
        Pd::Vector{Float64}, Qd::Vector{Float64},
        J0::SA.SparseMatrixCSC{Float64,Int64},
        Jpf::SA.SparseMatrixCSC{Float64,Int64},
        Hpf::Vector{SA.SparseMatrixCSC{Float64,Int64}},
        Hloss::SA.SparseMatrixCSC{Float64,Int64},
        mpc::Dict{String, Any},
        Yrec_mod::SA.SparseMatrixCSC{Float64,Int64},
        zYv::SA.SparseMatrixCSC{ComplexF64,Int64},
        vYz::SA.SparseMatrixCSC{ComplexF64,Int64},
        Yv::SA.SparseMatrixCSC{ComplexF64,Int64}}(
        (n, case.x0, u0, input.Pd, input.Qd,
        J0, Jpf, Hpf, Hloss, case.mpc, Yrec_mod,
        zYv, vYz, Yv))
    return qc_data, ind, c, opf_u, opf_x, opf_fobj, mpc
end

end
