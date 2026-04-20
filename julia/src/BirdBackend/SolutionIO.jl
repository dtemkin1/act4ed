function _bus_total_distance(data::BirdData, bus::BirdBus)
    total = 0.0
    current_node = data.depots[bus.depot]
    for (order, school_idx) in enumerate(bus.schools)
        route = data.routes[school_idx][bus.routes[order]]
        first_stop = data.stops[school_idx][route.stops[1]]
        total += travel_distance(data, current_node, first_stop)
        total += route_distance(data, school_idx, route)
        current_node = data.schools[school_idx]
    end
    total += travel_distance(data, current_node, data.depots[bus.depot])
    return total
end


function _bus_total_time(data::BirdData, bus::BirdBus)
    total = 0.0
    current_node = data.depots[bus.depot]
    for (order, school_idx) in enumerate(bus.schools)
        route = data.routes[school_idx][bus.routes[order]]
        first_stop = data.stops[school_idx][route.stops[1]]
        total += travel_time(data, current_node, first_stop)
        total += service_time(data, school_idx, route)
        current_node = data.schools[school_idx]
    end
    total += travel_time(data, current_node, data.depots[bus.depot])
    return total
end


function snapshot_solution(data::BirdData; runtime_seconds::Float64 = 0.0, status_name::String = "OPTIMAL", objective_value::Union{Nothing, Float64} = nothing)
    assignment_bus_ids = Int[]
    assignment_orders = Int[]
    assignment_school_indices = Int[]
    assignment_arrival_times = Float64[]
    assignment_distance_km = Float64[]
    assignment_service_time_min = Float64[]
    assignment_stop_ptr = Int[0]
    assignment_stop_values = Int[]

    for bus in data.buses
        current_node = data.depots[bus.depot]
        for (order, school_idx) in enumerate(bus.schools)
            route = data.routes[school_idx][bus.routes[order]]
            first_stop = data.stops[school_idx][route.stops[1]]
            route_distance_with_deadhead = travel_distance(data, current_node, first_stop) + route_distance(data, school_idx, route)
            push!(assignment_bus_ids, bus.id)
            push!(assignment_orders, order - 1)
            push!(assignment_school_indices, school_idx)
            push!(assignment_arrival_times, data.schools[school_idx].start_time - data.schools[school_idx].dwell_time)
            push!(assignment_distance_km, route_distance_with_deadhead)
            push!(assignment_service_time_min, service_time(data, school_idx, route))
            append!(assignment_stop_values, route.stops)
            push!(assignment_stop_ptr, length(assignment_stop_values))
            current_node = data.schools[school_idx]
        end
    end

    total_distance = sum(_bus_total_distance(data, bus) for bus in data.buses)
    total_time = sum(_bus_total_time(data, bus) for bus in data.buses)
    objective = objective_value === nothing ? float(length(data.buses)) : objective_value
    return BirdBackendSolution(
        status_name,
        objective,
        runtime_seconds,
        length(data.buses),
        total_distance,
        total_time,
        assignment_bus_ids,
        assignment_orders,
        assignment_school_indices,
        assignment_arrival_times,
        assignment_distance_km,
        assignment_service_time_min,
        assignment_stop_ptr,
        assignment_stop_values,
    )
end


function save_solution(path::AbstractString, solution::BirdBackendSolution)
    payload = Dict{String, Any}(
        "schema_version" => Int64(BIRD_SOLUTION_SCHEMA_VERSION),
        "status_name_utf8" => _encode_utf8_array(solution.status_name),
        "has_objective_value" => Int64(solution.objective_value === nothing ? 0 : 1),
        "objective_value" => Float64(solution.objective_value === nothing ? NaN : solution.objective_value),
        "runtime_seconds" => Float64(solution.runtime_seconds),
        "buses_used" => Int64(solution.buses_used),
        "total_distance_km" => Float64(solution.total_distance_km),
        "total_service_time_min" => Float64(solution.total_service_time_min),
        "assignment_bus_ids" => Int64.(solution.assignment_bus_ids),
        "assignment_orders" => Int64.(solution.assignment_orders),
        "assignment_school_indices" => Int64.(solution.assignment_school_indices),
        "assignment_arrival_times" => Float64.(solution.assignment_arrival_times),
        "assignment_distance_km" => Float64.(solution.assignment_distance_km),
        "assignment_service_time_min" => Float64.(solution.assignment_service_time_min),
        "assignment_stop_ptr" => Int64.(solution.assignment_stop_ptr),
        "assignment_stop_values" => Int64.(solution.assignment_stop_values),
    )
    NPZ.npzwrite(path, payload)
    return path
end
