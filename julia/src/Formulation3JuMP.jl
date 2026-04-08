module Formulation3JuMP

using Gurobi
using JuMP
using LinearAlgebra
using MathOptInterface
using NPZ
using SparseArrays

export Formulation3Instance, Formulation3Solution, load_instance, build_model, solve_problem!, snapshot_solution, save_solution

const MOI = MathOptInterface
const PORTABLE_SOLUTION_SCHEMA_VERSION = 1
const PORTABLE_VARIABLE_ARITIES = (
    z_b = 1,
    z_bq = 2,
    y_bqtau = 3,
    x_bqij = 3,
    v_bqi = 3,
    a_mbq = 3,
    T_bqi = 3,
    L_bqi = 3,
    e_bqs = 3,
    r_bmon = 1,
)


struct Formulation3Solution
    status::Int
    status_name::String
    objective_value::Union{Nothing, Float64}
    runtime_seconds::Float64
    variables::Dict{String, Tuple{Matrix{Int64}, Vector{Float64}}}
    meta::Dict{String, Any}
end


struct Formulation3Instance
    ALPHA::Float64
    BETA::Float64
    PHI::Float64
    EPSILON::Float64
    H_RIDE::Float64
    LAMBDA_ROUND::Float64
    M_TIME::Float64
    M_CAPACITY::Int
    T_horizon::Float64
    kappa_tau::Vector{Float64}

    nB::Int
    nM::Int
    nS::Int
    nP::Int
    nN::Int
    nA::Int
    nQ::Int
    nTau::Int

    arc_src::Vector{Int}
    arc_dst::Vector{Int}
    arc_time::Vector{Float64}
    arc_distance::Vector{Float64}

    pickup_node_p::Vector{Int}
    pickup_node_of_m::Vector{Int}
    school_node_of_m::Vector{Int}
    school_index_of_m::Vector{Int}
    tau_of_m::Vector{Int}
    is_flagged_m::Vector{Int}
    needs_wheelchair_m::Vector{Int}

    capacity_b::Vector{Float64}
    cap_upper_b::Vector{Float64}
    range_b::Vector{Float64}
    wheelchair_ok_b::Vector{Float64}
    start_depot_node_b::Vector{Int}
    end_depot_node_b::Vector{Int}

    school_node_s::Vector{Int}
    school_copy_node_s::Vector{Int}
    latest_arrival_s::Vector{Float64}

    node_out_arc::SparseMatrixCSC{Float64, Int}
    node_in_arc::SparseMatrixCSC{Float64, Int}
    bus_start_arc::SparseMatrixCSC{Float64, Int}
    bus_end_arc::SparseMatrixCSC{Float64, Int}
    school_copy_out_arc::SparseMatrixCSC{Float64, Int}
    school_copy_in_arc::SparseMatrixCSC{Float64, Int}
    student_to_pickup::SparseMatrixCSC{Float64, Int}
    student_to_school::SparseMatrixCSC{Float64, Int}
    student_to_school_type::SparseMatrixCSC{Float64, Int}

    student_to_pickup_node::SparseMatrixCSC{Float64, Int}
    student_to_school_node::SparseMatrixCSC{Float64, Int}
    arc_start_selector::SparseMatrixCSC{Float64, Int}
    arc_end_selector::SparseMatrixCSC{Float64, Int}
    node_arc_incidence::SparseMatrixCSC{Float64, Int}
    pickup_out_arc::SparseMatrixCSC{Float64, Int}
    pickup_in_arc::SparseMatrixCSC{Float64, Int}
    school_out_arc::SparseMatrixCSC{Float64, Int}
    school_in_arc::SparseMatrixCSC{Float64, Int}
    start_depot_selector::SparseMatrixCSC{Float64, Int}
    end_depot_selector::SparseMatrixCSC{Float64, Int}
    service_node_indices::Vector{Int}

    bus_names::Vector{String}
    student_names::Vector{String}
    node_names::Vector{String}
    node_ids::Vector{Int}
    school_ids::Vector{String}
end


_scalar(data, name, ::Type{T}) where {T} = T(data[name][])
_ivec(data, name) = Int.(vec(data[name]))
_fvec(data, name) = Float64.(vec(data[name]))
_svec(data, name) = [String(value) for value in vec(data[name])]
_default_names(prefix, n) = ["$(prefix)_$(i)" for i in 1:n]
_zero_affexpr_matrix(nrows, ncols) = [JuMP.AffExpr(0.0) for _ in 1:nrows, _ in 1:ncols]
_sum_expr(iter) = sum(iter; init = JuMP.AffExpr(0.0))
_encode_utf8_array(text::AbstractString) = collect(codeunits(String(text)))

function _status_code(status::MOI.TerminationStatusCode)::Int
    if status == MOI.OPTIMAL
        return 2
    elseif status == MOI.INFEASIBLE
        return 3
    elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
        return 4
    elseif status == MOI.UNBOUNDED
        return 5
    elseif status == MOI.TIME_LIMIT
        return 9
    elseif status == MOI.SOLUTION_LIMIT
        return 10
    elseif status == MOI.INTERRUPTED
        return 11
    elseif status == MOI.ALMOST_OPTIMAL
        return 13
    else
        return 0
    end
end

function _nonzero_snapshot(value::Float64)::Bool
    return abs(value) > 1.0e-9
end

function _snapshot_dense_container(container)::Tuple{Matrix{Int64}, Vector{Float64}}
    data = container isa JuMP.Containers.DenseAxisArray ? container.data : container
    index_rows = Vector{Vector{Int64}}()
    values = Float64[]
    for key in CartesianIndices(data)
        value = Float64(JuMP.value(data[key]))
        if !_nonzero_snapshot(value)
            continue
        end
        push!(index_rows, Int64[Tuple(key)...] .- 1)
        push!(values, value)
    end
    arity = ndims(data)
    if isempty(values)
        return zeros(Int64, 0, arity), Float64[]
    end
    indices = reduce(vcat, transpose.(index_rows))
    return indices, values
end

function _snapshot_sparse_container(
    container::JuMP.Containers.SparseAxisArray{JuMP.VariableRef, N},
)::Tuple{Matrix{Int64}, Vector{Float64}} where {N}
    index_rows = Vector{Vector{Int64}}()
    values = Float64[]
    arity = N
    for (key, var) in container.data
        value = Float64(JuMP.value(var))
        if !_nonzero_snapshot(value)
            continue
        end
        push!(index_rows, Int64[key...] .- 1)
        push!(values, value)
    end
    if isempty(values)
        return zeros(Int64, 0, arity), Float64[]
    end
    indices = reduce(vcat, transpose.(index_rows))
    return indices, values
end

function _snapshot_container(container, arity::Int)::Tuple{Matrix{Int64}, Vector{Float64}}
    if container isa JuMP.Containers.SparseAxisArray
        return _snapshot_sparse_container(container)
    end
    indices, values = _snapshot_dense_container(container)
    if size(indices, 2) != arity
        error("unexpected container arity $(size(indices, 2)); expected $(arity)")
    end
    return indices, values
end

function snapshot_solution(
    model::JuMP.Model,
    vars,
    meta = nothing,
)::Formulation3Solution
    status = termination_status(model)
    has_solution = has_values(model)
    variables = Dict{String, Tuple{Matrix{Int64}, Vector{Float64}}}()
    for (name, arity) in pairs(PORTABLE_VARIABLE_ARITIES)
        if has_solution
            variables[String(name)] = _snapshot_container(getproperty(vars, name), arity)
        else
            variables[String(name)] = (zeros(Int64, 0, arity), Float64[])
        end
    end
    meta_dict = Dict{String, Any}(
        "backend" => "julia",
    )
    if meta !== nothing
        if hasproperty(meta, :node_ids)
            meta_dict["node_ids"] = Int64.(meta.node_ids)
        end
        if hasproperty(meta, :arc_src)
            meta_dict["arc_src"] = Int64.(meta.arc_src .- 1)
        end
        if hasproperty(meta, :arc_dst)
            meta_dict["arc_dst"] = Int64.(meta.arc_dst .- 1)
        end
    end
    return Formulation3Solution(
        _status_code(status),
        string(status),
        has_solution ? Float64(objective_value(model)) : nothing,
        solve_time(model),
        variables,
        meta_dict,
    )
end

snapshot_solution(bundle::NamedTuple)::Formulation3Solution = snapshot_solution(bundle.model, bundle.vars, hasproperty(bundle, :meta) ? bundle.meta : nothing)

function save_solution(path::AbstractString, solution::Formulation3Solution)
    payload = Dict{String, Any}(
        "schema_version" => Int64(PORTABLE_SOLUTION_SCHEMA_VERSION),
        "status" => Int64(solution.status),
        "status_name_utf8" => _encode_utf8_array(solution.status_name),
        "has_objective_value" => Int64(isnothing(solution.objective_value) ? 0 : 1),
        "objective_value" => Float64(isnothing(solution.objective_value) ? NaN : solution.objective_value),
        "runtime_seconds" => Float64(solution.runtime_seconds),
        "meta_backend_utf8" => _encode_utf8_array(get(solution.meta, "backend", "julia")),
    )
    for (key, value) in solution.meta
        if value isa AbstractVector{<:Integer}
            payload["meta_$(key)"] = Int64.(value)
        end
    end
    for (name, _arity) in pairs(PORTABLE_VARIABLE_ARITIES)
        indices, values = solution.variables[String(name)]
        payload["$(name)_indices"] = indices
        payload["$(name)_values"] = values
    end
    NPZ.npzwrite(path, payload)
    return path
end

function save_solution(path::AbstractString, bundle::NamedTuple)
    return save_solution(path, snapshot_solution(bundle))
end

function _expand_block(container, q::Int, active_indices_by_q::Vector{Vector{Int}}, nrows::Int, ncols::Int)
    block = _zero_affexpr_matrix(nrows, ncols)
    for b in 1:nrows
        for idx in active_indices_by_q[q]
            JuMP.add_to_expression!(block[b, idx], 1.0, container[b, q, idx])
        end
    end
    return block
end

function _expand_block(container, q::Int, active_indices_by_bq::AbstractMatrix, nrows::Int, ncols::Int)
    block = _zero_affexpr_matrix(nrows, ncols)
    for b in 1:nrows
        for idx in active_indices_by_bq[b, q]
            JuMP.add_to_expression!(block[b, idx], 1.0, container[b, q, idx])
        end
    end
    return block
end

function _expand_assignment_block(container, q::Int, active_students_by_b::Vector{Vector{Int}}, nM::Int, nB::Int)
    block = _zero_affexpr_matrix(nM, nB)
    for b in 1:nB
        for m in active_students_by_b[b]
            JuMP.add_to_expression!(block[m, b], 1.0, container[m, b, q])
        end
    end
    return block
end

function _load_sparse(data, prefix)::SparseMatrixCSC{Float64, Int}
    nrows = _scalar(data, "$(prefix)_n_rows", Int)
    ncols = _scalar(data, "$(prefix)_n_cols", Int)
    rows = _ivec(data, "$(prefix)_rows")
    cols = _ivec(data, "$(prefix)_cols")
    vals = _fvec(data, "$(prefix)_vals")
    return sparse(rows, cols, vals, nrows, ncols)
end

function load_instance(path::AbstractString)::Formulation3Instance
    data = NPZ.npzread(
        path,
        [
            "ALPHA",
            "BETA",
            "PHI",
            "EPSILON",
            "H_RIDE",
            "LAMBDA_ROUND",
            "M_TIME",
            "M_CAPACITY",
            "T_horizon",
            "kappa_tau",
            "nB",
            "nM",
            "nS",
            "nP",
            "nN",
            "nA",
            "nQ",
            "nTau",
            "arc_src",
            "arc_dst",
            "arc_time",
            "arc_distance",
            "pickup_node_p",
            "pickup_node_of_m",
            "school_node_of_m",
            "school_index_of_m",
            "tau_of_m",
            "is_flagged_m",
            "needs_wheelchair_m",
            "capacity_b",
            "cap_upper_b",
            "range_b",
            "wheelchair_ok_b",
            "start_depot_node_b",
            "end_depot_node_b",
            "school_node_s",
            "school_copy_node_s",
            "latest_arrival_s",
            "node_out_arc_n_rows",
            "node_out_arc_n_cols",
            "node_out_arc_rows",
            "node_out_arc_cols",
            "node_out_arc_vals",
            "node_in_arc_n_rows",
            "node_in_arc_n_cols",
            "node_in_arc_rows",
            "node_in_arc_cols",
            "node_in_arc_vals",
            "bus_start_arc_n_rows",
            "bus_start_arc_n_cols",
            "bus_start_arc_rows",
            "bus_start_arc_cols",
            "bus_start_arc_vals",
            "bus_end_arc_n_rows",
            "bus_end_arc_n_cols",
            "bus_end_arc_rows",
            "bus_end_arc_cols",
            "bus_end_arc_vals",
            "school_copy_out_arc_n_rows",
            "school_copy_out_arc_n_cols",
            "school_copy_out_arc_rows",
            "school_copy_out_arc_cols",
            "school_copy_out_arc_vals",
            "school_copy_in_arc_n_rows",
            "school_copy_in_arc_n_cols",
            "school_copy_in_arc_rows",
            "school_copy_in_arc_cols",
            "school_copy_in_arc_vals",
            "student_to_pickup_n_rows",
            "student_to_pickup_n_cols",
            "student_to_pickup_rows",
            "student_to_pickup_cols",
            "student_to_pickup_vals",
            "student_to_school_n_rows",
            "student_to_school_n_cols",
            "student_to_school_rows",
            "student_to_school_cols",
            "student_to_school_vals",
            "student_to_school_type_n_rows",
            "student_to_school_type_n_cols",
            "student_to_school_type_rows",
            "student_to_school_type_cols",
            "student_to_school_type_vals",
            "node_ids",
        ],
    )

    nB = _scalar(data, "nB", Int)
    nM = _scalar(data, "nM", Int)
    nS = _scalar(data, "nS", Int)
    nP = _scalar(data, "nP", Int)
    nN = _scalar(data, "nN", Int)
    nA = _scalar(data, "nA", Int)
    nQ = _scalar(data, "nQ", Int)
    nTau = _scalar(data, "nTau", Int)

    arc_src = _ivec(data, "arc_src")
    arc_dst = _ivec(data, "arc_dst")
    pickup_node_of_m = _ivec(data, "pickup_node_of_m")
    school_node_of_m = _ivec(data, "school_node_of_m")
    pickup_node_p = _ivec(data, "pickup_node_p")
    school_node_s = _ivec(data, "school_node_s")
    school_copy_node_s = _ivec(data, "school_copy_node_s")
    start_depot_node_b = _ivec(data, "start_depot_node_b")
    end_depot_node_b = _ivec(data, "end_depot_node_b")

    student_to_pickup_node = sparse(
        pickup_node_of_m,
        collect(1:nM),
        ones(Float64, nM),
        nN,
        nM,
    )
    student_to_school_node = sparse(
        school_node_of_m,
        collect(1:nM),
        ones(Float64, nM),
        nN,
        nM,
    )
    arc_start_selector = sparse(
        arc_src,
        collect(1:nA),
        ones(Float64, nA),
        nN,
        nA,
    )
    arc_end_selector = sparse(
        arc_dst,
        collect(1:nA),
        ones(Float64, nA),
        nN,
        nA,
    )
    start_depot_selector = sparse(
        collect(1:nB),
        start_depot_node_b,
        ones(Float64, nB),
        nB,
        nN,
    )
    end_depot_selector = sparse(
        collect(1:nB),
        end_depot_node_b,
        ones(Float64, nB),
        nB,
        nN,
    )

    node_out_arc = _load_sparse(data, "node_out_arc")
    node_in_arc = _load_sparse(data, "node_in_arc")

    return Formulation3Instance(
        _scalar(data, "ALPHA", Float64),
        _scalar(data, "BETA", Float64),
        _scalar(data, "PHI", Float64),
        _scalar(data, "EPSILON", Float64),
        _scalar(data, "H_RIDE", Float64),
        _scalar(data, "LAMBDA_ROUND", Float64),
        _scalar(data, "M_TIME", Float64),
        _scalar(data, "M_CAPACITY", Int),
        _scalar(data, "T_horizon", Float64),
        _fvec(data, "kappa_tau"),
        nB,
        nM,
        nS,
        nP,
        nN,
        nA,
        nQ,
        nTau,
        arc_src,
        arc_dst,
        _fvec(data, "arc_time"),
        _fvec(data, "arc_distance"),
        pickup_node_p,
        pickup_node_of_m,
        school_node_of_m,
        _ivec(data, "school_index_of_m"),
        _ivec(data, "tau_of_m"),
        _ivec(data, "is_flagged_m"),
        _ivec(data, "needs_wheelchair_m"),
        _fvec(data, "capacity_b"),
        _fvec(data, "cap_upper_b"),
        _fvec(data, "range_b"),
        _fvec(data, "wheelchair_ok_b"),
        start_depot_node_b,
        end_depot_node_b,
        school_node_s,
        school_copy_node_s,
        _fvec(data, "latest_arrival_s"),
        node_out_arc,
        node_in_arc,
        _load_sparse(data, "bus_start_arc"),
        _load_sparse(data, "bus_end_arc"),
        _load_sparse(data, "school_copy_out_arc"),
        _load_sparse(data, "school_copy_in_arc"),
        _load_sparse(data, "student_to_pickup"),
        _load_sparse(data, "student_to_school"),
        _load_sparse(data, "student_to_school_type"),
        student_to_pickup_node,
        student_to_school_node,
        arc_start_selector,
        arc_end_selector,
        arc_end_selector - arc_start_selector,
        node_out_arc[pickup_node_p, :],
        node_in_arc[pickup_node_p, :],
        node_out_arc[school_node_s, :],
        node_in_arc[school_node_s, :],
        start_depot_selector,
        end_depot_selector,
        vcat(pickup_node_p, school_node_s),
        _default_names("bus", nB),
        _default_names("student", nM),
        _default_names("node", nN),
        _ivec(data, "node_ids"),
        _default_names("school", nS),
    )
end

function build_model(
    inst::Formulation3Instance;
    optimizer = Gurobi.Optimizer,
    log_file = nothing,
)
    model = optimizer === nothing ? Model() : Model(optimizer)
    if optimizer !== nothing && log_file !== nothing
        set_optimizer_attribute(model, "LogFile", String(log_file))
    end

    flagged_idx = findall(!iszero, inst.is_flagged_m)
    service_node_indices = sort(unique(inst.service_node_indices))
    service_node_mask = falses(inst.nN)
    service_node_mask[service_node_indices] .= true
    service_node_indices_by_q = [service_node_indices for _ in 1:inst.nQ]

    assignment_feasible = [
        iszero(inst.needs_wheelchair_m[m]) || !iszero(inst.wheelchair_ok_b[b]) for
        m in 1:inst.nM, b in 1:inst.nB
    ]
    active_student_indices_by_b = [
        Int[m for m in 1:inst.nM if assignment_feasible[m, b]] for b in 1:inst.nB
    ]
    active_bus_indices_by_m = [
        Int[b for b in 1:inst.nB if assignment_feasible[m, b]] for m in 1:inst.nM
    ]
    bus_has_flagged_candidate = [
        any(!iszero(inst.is_flagged_m[m]) && assignment_feasible[m, b] for m in 1:inst.nM)
        for b in 1:inst.nB
    ]

    school_copy_in_arc_set = Set(findnz(inst.school_copy_in_arc)[2])
    school_copy_out_arc_set = Set(findnz(inst.school_copy_out_arc)[2])
    active_arc_indices_by_q = Vector{Vector{Int}}(undef, inst.nQ)
    for q in 1:inst.nQ
        active_arc_indices_by_q[q] = [
            arc_idx for arc_idx in 1:inst.nA if !(
                arc_idx in school_copy_in_arc_set ||
                (q == 1 && arc_idx in school_copy_out_arc_set)
            )
        ]
    end

    time_active_nodes_by_bq = [Int[] for _ in 1:inst.nB, _ in 1:inst.nQ]
    load_active_nodes_by_bq = [Int[] for _ in 1:inst.nB, _ in 1:inst.nQ]
    school_copy_node_set = Set(inst.school_copy_node_s)
    for b in 1:inst.nB
        for q in 1:inst.nQ
            time_fixed_nodes = q == 1 ? Set([inst.start_depot_node_b[b]]) : Set{Int}()
            time_active_nodes_by_bq[b, q] = [
                node_idx for node_idx in 1:inst.nN if !(node_idx in time_fixed_nodes)
            ]

            load_fixed_nodes = copy(school_copy_node_set)
            push!(load_fixed_nodes, inst.end_depot_node_b[b])
            if q == 1
                push!(load_fixed_nodes, inst.start_depot_node_b[b])
            end
            load_active_nodes_by_bq[b, q] = [
                node_idx for node_idx in 1:inst.nN if !(node_idx in load_fixed_nodes)
            ]
        end
    end

    school_end_rounds = collect(1:max(inst.nQ - 1, 0))

    @variable(model, z_b[1:inst.nB], Bin)
    @variable(model, z_bq[1:inst.nB, 1:inst.nQ], Bin)
    @variable(model, y_bqtau[1:inst.nB, 1:inst.nQ, 1:inst.nTau], Bin)
    @variable(model, x_bqij[b = 1:inst.nB, q = 1:inst.nQ, a in active_arc_indices_by_q[q]], Bin)
    @variable(model, v_bqi[b = 1:inst.nB, q = 1:inst.nQ, n = 1:inst.nN; service_node_mask[n]], Bin)
    @variable(model, a_mbq[m = 1:inst.nM, b = 1:inst.nB, q = 1:inst.nQ; assignment_feasible[m, b]], Bin)
    @variable(
        model,
        0 <= T_bqi[
            b = 1:inst.nB,
            q = 1:inst.nQ,
            n in time_active_nodes_by_bq[b, q],
        ] <= inst.T_horizon,
    )
    @variable(
        model,
        L_bqi[
            b = 1:inst.nB,
            q = 1:inst.nQ,
            n in load_active_nodes_by_bq[b, q],
        ] >= 0,
        Int,
    )
    @variable(model, e_bqs[b = 1:inst.nB, q in school_end_rounds, s = 1:inst.nS], Bin)
    @variable(model, r_bmon[b = 1:inst.nB; bus_has_flagged_candidate[b]], Bin)
    r_bmon_full = [bus_has_flagged_candidate[b] ? r_bmon[b] : 0.0 for b in 1:inst.nB]

    @objective(
        model,
        Min,
        sum(
            inst.arc_distance[a] * x_bqij[b, q, a] for
            b in 1:inst.nB for q in 1:inst.nQ for a in active_arc_indices_by_q[q]
        ) +
        inst.LAMBDA_ROUND * sum(z_bq) +
        sum(r_bmon_full),
    )

    students_per_bus_round = [
        _sum_expr(a_mbq[m, b, q] for m in active_student_indices_by_b[b])
        for b in 1:inst.nB, q in 1:inst.nQ
    ]
    students_per_bus = [
        _sum_expr(students_per_bus_round[b, q] for q in 1:inst.nQ) for b in 1:inst.nB
    ]
    student_assignments = [
        _sum_expr(a_mbq[m, b, q] for b in active_bus_indices_by_m[m] for q in 1:inst.nQ)
        for m in 1:inst.nM
    ]

    @constraint(model, student_assignments .<= 1)
    @constraint(model, sum(a_mbq) >= inst.PHI * inst.nM)
    @constraint(
        model,
        [m = 1:inst.nM, b = 1:inst.nB, q = 1:inst.nQ; assignment_feasible[m, b]],
        a_mbq[m, b, q] <= z_bq[b, q],
    )
    @constraint(model, z_bq .<= students_per_bus_round)
    @constraint(model, z_b .<= students_per_bus)

    for q in 1:inst.nQ
        A_q = _expand_assignment_block(a_mbq, q, active_student_indices_by_b, inst.nM, inst.nB)
        X_q = _expand_block(x_bqij, q, active_arc_indices_by_q, inst.nB, inst.nA)
        V_q = _expand_block(v_bqi, q, service_node_indices_by_q, inst.nB, inst.nN)
        Y_q = y_bqtau[:, q, :]
        T_q = _expand_block(T_bqi, q, time_active_nodes_by_bq, inst.nB, inst.nN)
        L_q = _expand_block(L_bqi, q, load_active_nodes_by_bq, inst.nB, inst.nN)
        E_q = q < inst.nQ ? [e_bqs[b, q, s] for b in 1:inst.nB, s in 1:inst.nS] : zeros(Float64, inst.nB, inst.nS)

        pickup_assignments = transpose(inst.student_to_pickup * A_q)
        school_assignments = transpose(inst.student_to_school * A_q)
        pickup_node_assignments = transpose(inst.student_to_pickup_node * A_q)
        dropoff_node_assignments = transpose(inst.student_to_school_node * A_q)
        school_type_match = transpose(Y_q * inst.student_to_school_type)
        type_capacity_multiplier = Y_q * inst.kappa_tau

        start_from_depot = vec(sum(X_q .* inst.bus_start_arc, dims = 2))
        start_to_depot = vec(sum(X_q .* (inst.start_depot_selector * inst.node_in_arc), dims = 2))
        end_to_depot = vec(sum(X_q .* inst.bus_end_arc, dims = 2))
        end_from_depot = vec(sum(X_q .* (inst.end_depot_selector * inst.node_out_arc), dims = 2))
        school_copy_out = X_q * transpose(inst.school_copy_out_arc)
        pickup_out = X_q * transpose(inst.pickup_out_arc)
        pickup_in = X_q * transpose(inst.pickup_in_arc)
        school_out = X_q * transpose(inst.school_out_arc)
        school_in = X_q * transpose(inst.school_in_arc)

        pickup_visits_for_students = transpose(V_q[:, inst.pickup_node_of_m])
        school_visits_for_students = transpose(V_q[:, inst.school_node_of_m])
        school_times_for_students = transpose(T_q[:, inst.school_node_of_m])
        pickup_times_for_students = transpose(T_q[:, inst.pickup_node_of_m])

        time_deltas = T_q * inst.node_arc_incidence
        load_deltas = L_q * inst.node_arc_incidence
        pickup_at_arc_start = pickup_node_assignments * inst.arc_start_selector
        dropoff_at_arc_start = dropoff_node_assignments * inst.arc_start_selector
        load_change_at_arc_end =
            pickup_node_assignments * inst.arc_end_selector -
            dropoff_node_assignments * inst.arc_end_selector

        arc_time_rhs = ones(inst.nB) * transpose(inst.arc_time)
        latest_arrival_rhs = ones(inst.nB) * transpose(inst.latest_arrival_s)
        capacity_rhs = (inst.capacity_b .* type_capacity_multiplier) * ones(1, inst.nN)

        if q == 1
            @constraint(model, start_from_depot .== z_bq[:, q])
        else
            E_prev = [e_bqs[b, q - 1, s] for b in 1:inst.nB, s in 1:inst.nS]
            T_prev = _expand_block(T_bqi, q - 1, time_active_nodes_by_bq, inst.nB, inst.nN)
            A_prev = _expand_assignment_block(a_mbq, q - 1, active_student_indices_by_b, inst.nM, inst.nB)
            @constraint(model, start_from_depot .== 0)
            @constraint(model, school_copy_out .== E_prev)
            @constraint(
                model,
                T_q[:, inst.school_copy_node_s] .>=
                T_prev[:, inst.school_node_s] +
                inst.BETA .* transpose(inst.student_to_school * A_prev) -
                inst.M_TIME .* (1 .- E_prev),
            )
        end
        @constraint(model, start_to_depot .== 0)
        @constraint(model, end_from_depot .== 0)

        if q < inst.nQ
            @constraint(model, vec(sum(E_q, dims = 2)) .== z_bq[:, q + 1])
            @constraint(model, end_to_depot .== z_bq[:, q] - z_bq[:, q + 1])
        else
            @constraint(model, end_to_depot .== z_bq[:, q])
        end

        @constraint(model, pickup_out .== V_q[:, inst.pickup_node_p])
        @constraint(model, pickup_in .== V_q[:, inst.pickup_node_p])
        @constraint(model, V_q[:, inst.pickup_node_p] .<= pickup_assignments)
        @constraint(model, school_in .== V_q[:, inst.school_node_s])
        @constraint(model, school_out .== V_q[:, inst.school_node_s] .- E_q)
        @constraint(
            model,
            V_q[:, service_node_indices] .<= reshape(z_bq[:, q], inst.nB, 1),
        )
        @constraint(model, A_q .<= pickup_visits_for_students)
        @constraint(model, A_q .<= school_visits_for_students)

        @constraint(
            model,
            time_deltas .>=
            arc_time_rhs +
            inst.ALPHA .* pickup_at_arc_start +
            inst.BETA .* dropoff_at_arc_start -
            inst.M_TIME .* (1 .- X_q),
        )
        @constraint(
            model,
            T_q[:, inst.school_node_s] + inst.BETA .* school_assignments .<=
            latest_arrival_rhs + inst.M_TIME .* (1 .- V_q[:, inst.school_node_s]),
        )
        @constraint(
            model,
            school_times_for_students .>=
            pickup_times_for_students .+ inst.EPSILON .-
            inst.M_TIME .* (1 .- A_q),
        )
        @constraint(
            model,
            school_times_for_students - pickup_times_for_students .<=
            inst.H_RIDE .+ inst.M_TIME .* (1 .- A_q),
        )

        @constraint(
            model,
            load_deltas .>=
            load_change_at_arc_end -
            inst.M_CAPACITY .* (1 .- X_q),
        )
        @constraint(
            model,
            load_deltas .<=
            load_change_at_arc_end +
            inst.M_CAPACITY .* (1 .- X_q),
        )
        @constraint(model, L_q .<= capacity_rhs)
        @constraint(
            model,
            L_q[:, inst.school_node_s] .<=
            reshape(inst.cap_upper_b, inst.nB, 1) .* (1 .- E_q),
        )

        @constraint(model, vec(sum(Y_q, dims = 2)) .== z_bq[:, q])
        @constraint(model, A_q .<= school_type_match)
    end

    distance_per_bus = [
        sum(
            inst.arc_distance[a] * x_bqij[b, q, a] for
            q in 1:inst.nQ for a in active_arc_indices_by_q[q]
        ) for b in 1:inst.nB
    ]
    @constraint(
        model,
        distance_per_bus .<= inst.range_b .* z_b,
    )

    if !isempty(flagged_idx)
        flagged_totals = [
            _sum_expr(
                a_mbq[m, b, q] for
                m in flagged_idx for q in 1:inst.nQ if assignment_feasible[m, b]
            ) for b in 1:inst.nB
        ]
        for b in 1:inst.nB
            if !bus_has_flagged_candidate[b]
                continue
            end
            @constraint(model, r_bmon[b] <= flagged_totals[b])
            for m in flagged_idx
                if !assignment_feasible[m, b]
                    continue
                end
                for q in 1:inst.nQ
                    @constraint(model, r_bmon[b] >= a_mbq[m, b, q])
                end
            end
        end
    end

    vars = (
        z_b = z_b,
        z_bq = z_bq,
        y_bqtau = y_bqtau,
        x_bqij = x_bqij,
        v_bqi = v_bqi,
        a_mbq = a_mbq,
        T_bqi = T_bqi,
        L_bqi = L_bqi,
        e_bqs = e_bqs,
        r_bmon = r_bmon,
    )
    meta = (
        bus_names = inst.bus_names,
        student_names = inst.student_names,
        node_names = inst.node_names,
        node_ids = inst.node_ids,
        school_ids = inst.school_ids,
        arc_src = inst.arc_src,
        arc_dst = inst.arc_dst,
    )

    return (; model, vars, meta)
end

function solve_problem!(model::JuMP.Model)
    println("Solving MILP")
    optimize!(model)

    println()
    println("Status: ", termination_status(model))
    if has_values(model)
        println("Objective: ", objective_value(model))
    else
        println("Objective: unavailable")
    end
    return nothing
end

end
