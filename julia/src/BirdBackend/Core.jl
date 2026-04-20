const MOI = MathOptInterface
const BIRD_INSTANCE_SCHEMA_VERSION = 4
const BIRD_SOLUTION_SCHEMA_VERSION = 1
const DEFAULT_LAMBDA_VALUE = 1.0e4


struct BirdParameters
    bus_capacity::Int
    max_time_on_bus::Float64
    constant_stop_time::Float64
    stop_time_per_student::Float64
end


struct BirdSchool
    id::Int
    external_id::String
    name::String
    start_time::Float64
    dwell_time::Float64
    node_index::Int
end


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
end


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
    default_lambda_value::Float64
    scenarios::Vector{Vector{BirdScenario}}
    routes::Vector{Vector{BirdRoute}}
    used_scenario::Vector{Int}
    buses::Vector{BirdBus}
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


function _make_model(; optimizer = Gurobi.Optimizer, log_file = nothing, optimizer_attributes = Pair{String, Any}[])
    model = optimizer === nothing ? Model() : Model(optimizer)
    if optimizer !== nothing && log_file !== nothing
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


n_students(stop::BirdDemandStop) = stop.n_students
stop_time(data::BirdData, stop::BirdDemandStop) = data.params.constant_stop_time + data.params.stop_time_per_student * stop.n_students
max_travel_time(data::BirdData, _stop::BirdDemandStop) = data.params.max_time_on_bus


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
