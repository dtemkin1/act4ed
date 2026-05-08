const MOI = MathOptInterface
const BIRD_INSTANCE_SCHEMA_VERSION = 14
const BIRD_SOLUTION_SCHEMA_VERSION = 1
const DEFAULT_LAMBDA_VALUE = 1.0e4
const BIRD_TIMING_EPS = 1.0e-6
const SERVICE_GROUP_WHEELCHAIR = 1
const SERVICE_GROUP_SPED = 2
const SERVICE_GROUP_CONVENTIONAL = 3


struct BirdParameters
    bus_capacity::Int
    max_time_on_bus::Float64
    constant_stop_time::Float64
    stop_time_per_student::Float64
    stop_time_per_wheelchair_student::Float64
    stop_time_per_sped::Float64
end

BirdParameters(
    bus_capacity,
    max_time_on_bus,
    constant_stop_time,
    stop_time_per_student,
    stop_time_per_wheelchair_student,
) = BirdParameters(
    bus_capacity,
    max_time_on_bus,
    constant_stop_time,
    stop_time_per_student,
    stop_time_per_wheelchair_student,
    0.0,
)


struct BirdSchool
    id::Int
    external_id::String
    name::String
    start_time::Float64
    dwell_time::Float64
    earliest_arrival_buffer::Float64
    latest_arrival_buffer::Float64
    node_index::Int
end

BirdSchool(
    id::Int,
    external_id::String,
    name::String,
    start_time::Float64,
    dwell_time::Float64,
    node_index::Int,
) = BirdSchool(id, external_id, name, start_time, dwell_time, dwell_time, dwell_time, node_index)


struct BirdDepot
    id::Int
    external_id::String
    name::String
    node_index::Int
end


struct BirdDemandStop
    id::Int
    unique_id::Int
    external_id::String
    source_stop_id::String
    school_id::Int
    node_index::Int
    n_students::Int
    n_wheelchair::Int
    n_sped::Int
    group_id::Int
    grade_id::Int
end

BirdDemandStop(
    id,
    unique_id,
    external_id,
    source_stop_id,
    school_id,
    node_index,
    n_students,
    n_wheelchair,
    group_id,
    grade_id,
) = BirdDemandStop(
    id,
    unique_id,
    external_id,
    source_stop_id,
    school_id,
    node_index,
    n_students,
    n_wheelchair,
    0,
    group_id,
    grade_id,
)


struct BirdScenario
    school::Int
    id::Int
    route_ids::Vector{Int}
end


struct BirdRoute
    id::Int
    stops::Vector{Int}
end


struct BirdBus
    id::Int
    depot::Int
    schools::Vector{Int}
    routes::Vector{Int}
    arrival_times::Vector{Float64}
end

BirdBus(id::Int, depot::Int, schools::Vector{Int}, routes::Vector{Int}) =
    BirdBus(id, depot, schools, routes, Float64[])


struct BirdFleetBus
    id::Int
    name::String
    depot::Int
    capacity::Int
    has_monitor::Bool
    wheelchair_capacity::Int
    bus_type::String
end


mutable struct BirdData
    params::BirdParameters
    schools::Vector{BirdSchool}
    depots::Vector{BirdDepot}
    stops::Vector{Vector{BirdDemandStop}}
    travel_time_min::Matrix{Float64}
    travel_distance_km::Matrix{Float64}
    bus_capacity::Int
    fleet_size::Int
    cohort::String
    bus_type::String
    fleet_aware::Bool
    conventional_spillover::Bool
    allow_partial::Bool
    fleet::Vector{BirdFleetBus}
    default_lambda_value::Float64
    scenarios::Vector{Vector{BirdScenario}}
    routes::Vector{Vector{BirdRoute}}
    used_scenario::Vector{Int}
    buses::Vector{BirdBus}
    unassigned_stops::Vector{Tuple{Int, Int}}
    method::String
end


struct BirdScenarioParameters
    max_route_time_lower::Float64
    max_route_time_upper::Float64
    n_greedy::Int
    lambda_value::Float64
    n_iterations::Int
end


struct BirdBackendSolution
    status_name::String
    objective_value::Union{Nothing, Float64}
    runtime_seconds::Float64
    buses_used::Int
    total_distance_km::Float64
    total_service_time_min::Float64
    assignment_bus_ids::Vector{Int}
    assignment_orders::Vector{Int}
    assignment_school_indices::Vector{Int}
    assignment_arrival_times::Vector{Float64}
    assignment_distance_km::Vector{Float64}
    assignment_service_time_min::Vector{Float64}
    assignment_stop_ptr::Vector{Int}
    assignment_stop_values::Vector{Int}
    unassigned_school_indices::Vector{Int}
    unassigned_stop_indices::Vector{Int}
end


struct DirectedEdge
    src::Int
    dst::Int
end


mutable struct DirectedGraph
    out_neighbors::Vector{Vector{Int}}
    in_neighbors::Vector{Vector{Int}}
end


DirectedGraph(n::Int) = DirectedGraph([Int[] for _ in 1:n], [Int[] for _ in 1:n])


function add_edge!(graph::DirectedGraph, src::Int, dst::Int)
    dst in graph.out_neighbors[src] && return false
    push!(graph.out_neighbors[src], dst)
    push!(graph.in_neighbors[dst], src)
    return true
end


edges(graph::DirectedGraph) = (
    DirectedEdge(src, dst) for src in eachindex(graph.out_neighbors) for dst in graph.out_neighbors[src]
)

edges_out(graph::DirectedGraph, node::Int) = (DirectedEdge(node, dst) for dst in graph.out_neighbors[node])
edges_in(graph::DirectedGraph, node::Int) = (DirectedEdge(src, node) for src in graph.in_neighbors[node])


function _scalar(data, name::AbstractString, ::Type{T}) where {T}
    return T(data[name][])
end


_ivec(data, name::AbstractString) = Int.(vec(data[name]))
_fvec(data, name::AbstractString) = Float64.(vec(data[name]))


function _decode_string_array(data, name::AbstractString)
    values = vec(data[name])
    decoded = String[]
    for value in values
        if value isa AbstractString
            push!(decoded, String(value))
        else
            push!(decoded, String(UInt8.(vec(value))))
        end
    end
    return decoded
end


_encode_utf8_array(value::AbstractString) = collect(codeunits(String(value)))


function _log_timing(label::AbstractString, enabled::Bool, elapsed_seconds::Real)
    if enabled
        println(stderr, "[bird timing] $(label): $(round(elapsed_seconds; digits = 3))s")
        flush(stderr)
    end
end


function _log_timing_message(message::AbstractString, enabled::Bool)
    if enabled
        println(stderr, "[bird timing] $(message)")
        flush(stderr)
    end
end


function _timed_value(label::AbstractString, enabled::Bool, f::Function)
    start_time = time()
    value = f()
    _log_timing(label, enabled, time() - start_time)
    return value
end


_timed_value(f::Function, label::AbstractString, enabled::Bool) = _timed_value(label, enabled, f)


function _make_model(; optimizer = Gurobi.Optimizer, log_file = nothing, optimizer_attributes = Pair{String, Any}[])
    optimizer === nothing && return Model()

    if optimizer === Gurobi.Optimizer
        params = Dict{String, Any}()
        if log_file !== nothing
            params["LogFile"] = String(log_file)
        end
        for (name, value) in optimizer_attributes
            params[String(name)] = value
        end
        attribute_names = Set(keys(params))
        has_log_file = "LogFile" in attribute_names
        has_log_control = "OutputFlag" in attribute_names || "LogToConsole" in attribute_names
        if !has_log_control
            if has_log_file
                params["LogToConsole"] = 0
            else
                params["OutputFlag"] = 0
            end
        end
        env = Gurobi.Env(params)
        return Model(() -> Gurobi.Optimizer(env))
    end

    model = Model(optimizer)
    if log_file !== nothing
        set_optimizer_attribute(model, "LogFile", String(log_file))
    end
    for (name, value) in optimizer_attributes
        set_optimizer_attribute(model, name, value)
    end
    return model
end


travel_time(data::BirdData, src::Int, dst::Int) = data.travel_time_min[src, dst]
travel_distance(data::BirdData, src::Int, dst::Int) = data.travel_distance_km[src, dst]
travel_time(data::BirdData, src::BirdDepot, dst::BirdDemandStop) = travel_time(data, src.node_index, dst.node_index)
travel_time(data::BirdData, src::BirdSchool, dst::BirdDemandStop) = travel_time(data, src.node_index, dst.node_index)
travel_time(data::BirdData, src::BirdDemandStop, dst::BirdDemandStop) = travel_time(data, src.node_index, dst.node_index)
travel_time(data::BirdData, src::BirdDemandStop, dst::BirdSchool) = travel_time(data, src.node_index, dst.node_index)
travel_time(data::BirdData, src::BirdSchool, dst::BirdDepot) = travel_time(data, src.node_index, dst.node_index)
travel_time(data::BirdData, src::BirdDemandStop, dst::BirdDepot) = travel_time(data, src.node_index, dst.node_index)
travel_distance(data::BirdData, src::BirdDepot, dst::BirdDemandStop) = travel_distance(data, src.node_index, dst.node_index)
travel_distance(data::BirdData, src::BirdSchool, dst::BirdDemandStop) = travel_distance(data, src.node_index, dst.node_index)
travel_distance(data::BirdData, src::BirdDemandStop, dst::BirdDemandStop) = travel_distance(data, src.node_index, dst.node_index)
travel_distance(data::BirdData, src::BirdDemandStop, dst::BirdSchool) = travel_distance(data, src.node_index, dst.node_index)
travel_distance(data::BirdData, src::BirdSchool, dst::BirdDepot) = travel_distance(data, src.node_index, dst.node_index)


earliest_arrival_time(school::BirdSchool) = school.start_time - school.earliest_arrival_buffer
latest_arrival_time(school::BirdSchool) = school.start_time - school.latest_arrival_buffer
uses_original_dwell_timing(school::BirdSchool) =
    abs(school.earliest_arrival_buffer - school.dwell_time) <= BIRD_TIMING_EPS &&
    abs(school.latest_arrival_buffer - school.dwell_time) <= BIRD_TIMING_EPS
uses_original_dwell_timing(data::BirdData) = all(uses_original_dwell_timing, data.schools)
n_students(stop::BirdDemandStop) = stop.n_students
function stop_time(data::BirdData, stop::BirdDemandStop)
    overlap = min(stop.n_wheelchair, stop.n_sped)
    wheelchair_only = stop.n_wheelchair - overlap
    sped_only = stop.n_sped - overlap
    return (
        data.params.constant_stop_time +
        data.params.stop_time_per_student * stop.n_students +
        data.params.stop_time_per_wheelchair_student * wheelchair_only +
        data.params.stop_time_per_sped * sped_only +
        max(data.params.stop_time_per_wheelchair_student, data.params.stop_time_per_sped) * overlap
    )
end
max_travel_time(data::BirdData, _stop::BirdDemandStop) = data.params.max_time_on_bus
route_grade_id(data::BirdData, school_idx::Int, route::BirdRoute) =
    isempty(route.stops) ? 0 : data.stops[school_idx][route.stops[1]].grade_id


function _school_start_time(text::AbstractString)
    raw = strip(text)
    hours = parse(Int, raw[1:end-2])
    minutes = parse(Int, raw[end-1:end])
    return 60.0 * hours + minutes
end


function _read_tsv(path::AbstractString)
    lines = readlines(path)
    header = split(chomp(lines[1]), '\t')
    rows = Vector{Dict{String, String}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(chomp(line), '\t')
        row = Dict{String, String}()
        for (key, value) in zip(header, values)
            row[key] = value
        end
        push!(rows, row)
    end
    return rows
end
