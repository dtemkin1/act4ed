abstract type ScenarioNode end


struct ArrivalNode <: ScenarioNode
    id::Int
    school::Int
    route::BirdRoute
    service_time::Float64
    scenario::Int
end


struct DepartureNode <: ScenarioNode
    id::Int
    school::Int
    capacity::Int
    scenario::Int
end


struct YardNode <: ScenarioNode
    id::Int
    yard::Int
end


mutable struct ScenarioGraph
    graph::DirectedGraph
    nodes::Vector{ScenarioNode}
    node_types::Dict{Symbol, Vector{Int}}
    costs::Dict{Tuple{Int, Int}, Float64}
    num_scenarios::Vector{Int}
end


function get_yard_nodes!(data::BirdData, node_types::Dict{Symbol, Vector{Int}})
    nodes = ScenarioNode[]
    node_id = 1
    for depot in data.depots
        push!(nodes, YardNode(node_id, depot.id))
        push!(node_types[:yard], node_id)
        node_id += 1
    end
    return nodes, node_id
end


function get_school_scenario_nodes!(data::BirdData, school_idx::Int, current_node_id::Int, scenario_num::Int, node_types::Dict{Symbol, Vector{Int}})
    nodes = ScenarioNode[]
    scenario = data.scenarios[school_idx][scenario_num]
    for route_id in scenario.route_ids
        route = data.routes[school_idx][route_id]
        push!(nodes, ArrivalNode(current_node_id, school_idx, route, service_time(data, school_idx, route), scenario.id))
        push!(node_types[:arrival], current_node_id)
        current_node_id += 1
    end
    push!(nodes, DepartureNode(current_node_id, school_idx, length(scenario.route_ids), scenario_num))
    push!(node_types[:departure], current_node_id)
    return nodes, current_node_id + 1
end


function is_feasible_in_time(data::BirdData, school1::Int, school2::Int, route::BirdRoute, route_time::Float64)
    first_stop = data.stops[school2][route.stops[1]]
    start_school = data.schools[school1]
    end_school = data.schools[school2]
    transfer_time = travel_time(data, start_school.node_index, first_stop.node_index)
    isfinite(transfer_time) || return false
    return start_school.start_time + transfer_time + route_time + end_school.dwell_time <= end_school.start_time
end


function create_edge!(data::BirdData, graph::DirectedGraph, node1::ScenarioNode, node2::ScenarioNode)
    return false, 0.0
end


function create_edge!(data::BirdData, graph::DirectedGraph, node1::DepartureNode, node2::ArrivalNode)
    if node1.school != node2.school && is_feasible_in_time(data, node1.school, node2.school, node2.route, node2.service_time)
        return add_edge!(graph, node1.id, node2.id), 0.0
    end
    return false, 0.0
end


create_edge!(data::BirdData, graph::DirectedGraph, node1::DepartureNode, node2::YardNode) = (add_edge!(graph, node1.id, node2.id), 0.0)


function create_edge!(data::BirdData, graph::DirectedGraph, node1::ArrivalNode, node2::DepartureNode)
    if node1.school == node2.school && node1.scenario == node2.scenario
        return add_edge!(graph, node1.id, node2.id), 0.0
    end
    return false, 0.0
end


create_edge!(data::BirdData, graph::DirectedGraph, node1::YardNode, node2::ArrivalNode) = (add_edge!(graph, node1.id, node2.id), 1.0)


function build_scenario_graph(data::BirdData)
    nodes = ScenarioNode[]
    node_types = Dict(:yard => Int[], :arrival => Int[], :departure => Int[])
    yard_nodes, current_node_id = get_yard_nodes!(data, node_types)
    append!(nodes, yard_nodes)
    num_scenarios = [length(data.scenarios[idx]) for idx in eachindex(data.schools)]
    for school_idx in eachindex(data.schools)
        for scenario_num in 1:num_scenarios[school_idx]
            school_nodes, current_node_id = get_school_scenario_nodes!(data, school_idx, current_node_id, scenario_num, node_types)
            append!(nodes, school_nodes)
        end
    end
    graph = DirectedGraph(length(nodes))
    costs = Dict{Tuple{Int, Int}, Float64}()
    for type1 in (:departure, :arrival, :yard)
        for type2 in (:departure, :arrival, :yard)
            for idx1 in node_types[type1], idx2 in node_types[type2]
                added, cost = create_edge!(data, graph, nodes[idx1], nodes[idx2])
                if added
                    costs[(idx1, idx2)] = cost
                end
            end
        end
    end
    return ScenarioGraph(graph, nodes, node_types, costs, num_scenarios)
end


function select_scenario(data::BirdData, scenario_graph::ScenarioGraph; optimizer = Gurobi.Optimizer, optimizer_attributes = Pair{String, Any}[])
    model = _make_model(; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    @variable(model, use_scenario[school in eachindex(data.schools), scenario in 1:scenario_graph.num_scenarios[school]], Bin)
    @variable(model, bus_flow[edge in edges(scenario_graph.graph)] >= 0, Int)

    @constraint(
        model,
        [node_id in scenario_graph.node_types[:departure]],
        sum(bus_flow[edge] for edge in edges_out(scenario_graph.graph, node_id)) <= scenario_graph.nodes[node_id].capacity,
    )
    @constraint(
        model,
        [node_id in scenario_graph.node_types[:arrival]],
        sum(bus_flow[edge] for edge in edges_out(scenario_graph.graph, node_id)) ==
        use_scenario[scenario_graph.nodes[node_id].school, scenario_graph.nodes[node_id].scenario],
    )
    @constraint(model, [school in eachindex(data.schools)], sum(use_scenario[school, scenario] for scenario in 1:scenario_graph.num_scenarios[school]) == 1)
    @constraint(
        model,
        [node_id in eachindex(scenario_graph.nodes)],
        sum(bus_flow[edge] for edge in edges_in(scenario_graph.graph, node_id)) ==
        sum(bus_flow[edge] for edge in edges_out(scenario_graph.graph, node_id)),
    )
    @objective(model, Min, sum(bus_flow[edge] * get(scenario_graph.costs, (edge.src, edge.dst), 0.0) for edge in edges(scenario_graph.graph)))
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("scenario selection failed with $(status)")
    return [argmax([value(use_scenario[school, scenario]) for scenario in 1:scenario_graph.num_scenarios[school]]) for school in eachindex(data.schools)]
end


abstract type FullRoutingNode end


struct BusNode <: FullRoutingNode
    id::Int
    depot_id::Int
    route::BirdRoute
    school::Int
    service_time::Float64
end


struct FullYardNode <: FullRoutingNode
    id::Int
    depot_id::Int
end


mutable struct FullRoutingGraph
    graph::DirectedGraph
    nodes::Vector{FullRoutingNode}
    costs::Dict{Tuple{Int, Int}, Float64}
end


function build_full_routing_graph(data::BirdData, used_scenario::Vector{Int})
    nodes = FullRoutingNode[]
    current_id = 1
    for depot in data.depots
        push!(nodes, FullYardNode(current_id, depot.id))
        current_id += 1
    end
    for school_idx in eachindex(data.schools)
        scenario = data.scenarios[school_idx][used_scenario[school_idx]]
        for route_id in scenario.route_ids
            route = data.routes[school_idx][route_id]
            for depot in data.depots
                push!(nodes, BusNode(current_id, depot.id, route, school_idx, service_time(data, school_idx, route)))
                current_id += 1
            end
        end
    end
    graph = DirectedGraph(length(nodes))
    costs = Dict{Tuple{Int, Int}, Float64}()
    for node1 in nodes, node2 in nodes
        added, cost = create_edge!(data, graph, node1, node2)
        if added
            costs[(node1.id, node2.id)] = cost
        end
    end
    return FullRoutingGraph(graph, nodes, costs)
end


function create_edge!(data::BirdData, graph::DirectedGraph, node1::FullRoutingNode, node2::FullRoutingNode)
    return false, 0.0
end


function create_edge!(data::BirdData, graph::DirectedGraph, node1::FullYardNode, node2::BusNode)
    if node1.depot_id == node2.depot_id
        first_stop = data.stops[node2.school][node2.route.stops[1]]
        depot = data.depots[node1.depot_id]
        return add_edge!(graph, node1.id, node2.id), travel_time(data, depot, first_stop)
    end
    return false, 0.0
end


function create_edge!(data::BirdData, graph::DirectedGraph, node1::BusNode, node2::FullYardNode)
    if node1.depot_id == node2.depot_id
        school = data.schools[node1.school]
        depot = data.depots[node2.depot_id]
        return add_edge!(graph, node1.id, node2.id), travel_time(data, school, depot)
    end
    return false, 0.0
end


function create_edge!(data::BirdData, graph::DirectedGraph, node1::BusNode, node2::BusNode)
    if node1.depot_id == node2.depot_id && node1.school != node2.school && is_feasible_in_time(data, node1.school, node2.school, node2.route, node2.service_time)
        school = data.schools[node1.school]
        first_stop = data.stops[node2.school][node2.route.stops[1]]
        return add_edge!(graph, node1.id, node2.id), travel_time(data, school, first_stop)
    end
    return false, 0.0
end


function yard_to_node_dict(graph::FullRoutingGraph)
    result = Dict{Int, Int}()
    for node in graph.nodes
        if node isa FullYardNode
            result[node.depot_id] = node.id
        end
    end
    return result
end


function solve_full_routing(data::BirdData, graph::FullRoutingGraph; optimizer = Gurobi.Optimizer, optimizer_attributes = Pair{String, Any}[])
    model = _make_model(; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    @variable(model, bus_flow[edge in edges(graph.graph)], Bin)
    @variable(model, yard_capacity[node_id in eachindex(graph.nodes); graph.nodes[node_id] isa FullYardNode] >= 0, Int)
    @constraint(
        model,
        [node_id in eachindex(graph.nodes); graph.nodes[node_id] isa BusNode],
        sum(bus_flow[edge] for edge in edges_in(graph.graph, node_id)) == 1,
    )
    @constraint(
        model,
        [node_id in eachindex(graph.nodes)],
        sum(bus_flow[edge] for edge in edges_in(graph.graph, node_id)) ==
        sum(bus_flow[edge] for edge in edges_out(graph.graph, node_id)),
    )
    @constraint(
        model,
        [node_id in eachindex(graph.nodes); graph.nodes[node_id] isa FullYardNode],
        sum(bus_flow[edge] for edge in edges_out(graph.graph, node_id)) <= yard_capacity[node_id],
    )
    @objective(model, Min, sum(yard_capacity[node_id] for node_id in eachindex(graph.nodes) if graph.nodes[node_id] isa FullYardNode))
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("full routing solve failed with $(status)")
    flows = Dict(edge => value(bus_flow[edge]) for edge in edges(graph.graph))
    return interpret_flows!(data, graph, flows)
end


function follow_bus_along_route!(graph::FullRoutingGraph, flows::Dict{DirectedEdge, Float64}, yard_node::Int)
    schools = Int[]
    routes = Int[]
    current_node = yard_node
    while true
        found = false
        for edge in edges_out(graph.graph, current_node)
            if get(flows, edge, 0.0) > 0.5
                found = true
                dst_node = graph.nodes[edge.dst]
                if dst_node isa BusNode
                    push!(schools, dst_node.school)
                    push!(routes, dst_node.route.id)
                    current_node = edge.dst
                else
                    flows[edge] = 0.0
                    return schools, routes
                end
                flows[edge] = 0.0
                break
            end
        end
        found || return schools, routes
    end
end


function interpret_flows!(data::BirdData, graph::FullRoutingGraph, flows::Dict{DirectedEdge, Float64})
    final_buses = BirdBus[]
    yard_nodes = yard_to_node_dict(graph)
    current_bus_id = 1
    for depot in data.depots
        yard_node = yard_nodes[depot.id]
        while true
            schools, routes = follow_bus_along_route!(graph, flows, yard_node)
            isempty(schools) && break
            push!(final_buses, BirdBus(current_bus_id, depot.id, schools, routes))
            current_bus_id += 1
        end
    end
    return final_buses
end


function route_buses!(data::BirdData; optimizer = Gurobi.Optimizer, optimizer_attributes = Pair{String, Any}[])
    scenario_graph = build_scenario_graph(data)
    data.used_scenario = select_scenario(data, scenario_graph; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    full_routing_graph = build_full_routing_graph(data, data.used_scenario)
    data.buses = solve_full_routing(data, full_routing_graph; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    return data
end
