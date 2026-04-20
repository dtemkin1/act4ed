struct GreedyState
    route::BirdRoute
    n_students::Int
    slack_times::Vector{Float64}
    stop_times::Vector{Float64}
    route_time::Float64
end


function initial_route(data::BirdData, school_idx::Int, stop_idx::Int, route_id::Int)
    stop = data.stops[school_idx][stop_idx]
    school = data.schools[school_idx]
    time_on_bus = travel_time(data, stop, school)
    slack_times = [max_travel_time(data, stop) - time_on_bus]
    stop_times = [time_on_bus]
    route_time = time_on_bus + stop_time(data, stop)
    return GreedyState(BirdRoute(route_id, [stop_idx]), stop.n_students, slack_times, stop_times, route_time)
end


function best_insertion(data::BirdData, school_idx::Int, new_stop_idx::Int, state::GreedyState, max_route_time::Float64)
    new_stop = data.stops[school_idx][new_stop_idx]
    school = data.schools[school_idx]
    best_time_diff = Inf
    insert_id = -1

    time_diff = travel_time(data, new_stop, data.stops[school_idx][state.route.stops[1]])
    total_time = time_diff + state.route_time + stop_time(data, new_stop)
    if total_time - stop_time(data, new_stop) <= max_travel_time(data, new_stop) && total_time <= max_route_time
        best_time_diff = time_diff
        insert_id = 0
    end

    for idx in 1:(length(state.route.stops) - 1)
        next_stop = data.stops[school_idx][state.route.stops[idx + 1]]
        current_stop = data.stops[school_idx][state.route.stops[idx]]
        time_to_next = travel_time(data, new_stop, next_stop)
        time_diff =
            travel_time(data, current_stop, new_stop) +
            time_to_next -
            travel_time(data, current_stop, next_stop)
        if time_diff < best_time_diff
            total_time = time_diff + state.route_time + stop_time(data, new_stop)
            feasible =
                time_diff + stop_time(data, new_stop) <= state.slack_times[idx] &&
                state.stop_times[idx + 1] + time_to_next + stop_time(data, next_stop) <= max_travel_time(data, new_stop) &&
                total_time <= max_route_time
            if feasible
                best_time_diff = time_diff
                insert_id = idx
            end
        end
    end

    previous_stop = data.stops[school_idx][state.route.stops[end]]
    time_diff =
        travel_time(data, previous_stop, new_stop) +
        travel_time(data, new_stop, school) -
        travel_time(data, previous_stop, school)
    if time_diff < best_time_diff
        total_time = time_diff + state.route_time + stop_time(data, new_stop)
        feasible =
            time_diff + stop_time(data, new_stop) <= state.slack_times[end] &&
            travel_time(data, new_stop, school) <= max_travel_time(data, new_stop) &&
            total_time <= max_route_time
        if feasible
            best_time_diff = time_diff
            insert_id = length(state.route.stops)
        end
    end
    return insert_id, best_time_diff
end


function build_route(data::BirdData, state::GreedyState, school_idx::Int, new_stop_idx::Int, insert_id::Int)
    new_stop = data.stops[school_idx][new_stop_idx]
    school = data.schools[school_idx]
    if insert_id == 0
        next_stop = data.stops[school_idx][state.route.stops[1]]
        new_stop_time = travel_time(data, new_stop, next_stop) + state.stop_times[1] + stop_time(data, next_stop)
        time_diff = 0.0
    elseif insert_id == length(state.route.stops)
        previous_stop = data.stops[school_idx][state.route.stops[end]]
        new_stop_time = travel_time(data, new_stop, school)
        time_diff =
            new_stop_time + travel_time(data, previous_stop, new_stop) - travel_time(data, previous_stop, school)
    else
        previous_stop = data.stops[school_idx][state.route.stops[insert_id]]
        next_stop = data.stops[school_idx][state.route.stops[insert_id + 1]]
        time_to_next = travel_time(data, new_stop, next_stop)
        time_diff =
            travel_time(data, previous_stop, new_stop) +
            time_to_next -
            travel_time(data, previous_stop, next_stop)
        new_stop_time = time_to_next + state.stop_times[insert_id + 1] + stop_time(data, next_stop)
    end

    new_stop_times = vcat(
        state.stop_times[1:insert_id] .+ stop_time(data, new_stop) .+ time_diff,
        [new_stop_time],
        state.stop_times[(insert_id + 1):end],
    )
    new_stops = vcat(state.route.stops[1:insert_id], [new_stop_idx], state.route.stops[(insert_id + 1):end])
    new_slack = vcat(
        state.slack_times[1:insert_id] .- stop_time(data, new_stop) .- time_diff,
        [max_travel_time(data, new_stop) - new_stop_time],
        state.slack_times[(insert_id + 1):end],
    )
    for idx in eachindex(new_slack)
        if idx > 1
            new_slack[idx] = min(new_slack[idx - 1], new_slack[idx])
        end
    end
    route_time = new_stop_times[1] + stop_time(data, data.stops[school_idx][new_stops[1]])
    return GreedyState(
        BirdRoute(state.route.id, new_stops),
        state.n_students + new_stop.n_students,
        new_slack,
        new_stop_times,
        route_time,
    )
end


function greedy_routes(data::BirdData, school_idx::Int, max_route_time::Float64; rng = Random.default_rng())
    routes = BirdRoute[]
    available = trues(length(data.stops[school_idx]))
    while any(available)
        start_options = findall(identity, available)
        start_stop_idx = rand(rng, start_options)
        state = initial_route(data, school_idx, start_stop_idx, length(routes) + 1)
        available[start_stop_idx] = false
        while true
            best_stop_idx = 0
            best_insert_idx = -1
            best_time_diff = Inf
            for stop_idx in findall(identity, available)
                if data.stops[school_idx][stop_idx].n_students + state.n_students <= data.params.bus_capacity
                    insert_idx, time_diff = best_insertion(data, school_idx, stop_idx, state, max_route_time)
                    if time_diff < best_time_diff
                        best_time_diff = time_diff
                        best_stop_idx = stop_idx
                        best_insert_idx = insert_idx
                    end
                end
            end
            if isfinite(best_time_diff)
                state = build_route(data, state, school_idx, best_stop_idx, best_insert_idx)
                available[best_stop_idx] = false
            else
                push!(routes, state.route)
                break
            end
        end
    end
    return routes
end


function service_time(data::BirdData, school_idx::Int, stop_list::Vector{Int})
    stops = data.stops[school_idx]
    school = data.schools[school_idx]
    t = travel_time(data, stops[stop_list[end]], school)
    for idx in length(stop_list):-1:2
        t += travel_time(data, stops[stop_list[idx - 1]], stops[stop_list[idx]]) + stop_time(data, stops[stop_list[idx]])
    end
    t += stop_time(data, stops[stop_list[1]])
    return t
end


service_time(data::BirdData, school_idx::Int, route::BirdRoute) = service_time(data, school_idx, route.stops)


function route_distance(data::BirdData, school_idx::Int, route::BirdRoute)
    stops = data.stops[school_idx]
    school = data.schools[school_idx]
    total = travel_distance(data, stops[route.stops[end]], school)
    for idx in length(route.stops):-1:2
        total += travel_distance(data, stops[route.stops[idx - 1]], stops[route.stops[idx]])
    end
    return total
end


function sum_individual_travel_times(data::BirdData, school_idx::Int, route::BirdRoute)
    stops = data.stops[school_idx]
    num_stops = length(route.stops)
    total = travel_time(data, stops[route.stops[end]], data.schools[school_idx]) * num_stops
    while num_stops > 1
        total +=
            (
                travel_time(data, stops[route.stops[num_stops - 1]], stops[route.stops[num_stops]]) +
                stop_time(data, stops[route.stops[num_stops]])
            ) * (num_stops - 1)
        num_stops -= 1
    end
    return total
end


struct FeasibleRoute
    stop_ids::Vector{Int}
    cost::Float64
end


FeasibleRoute(data::BirdData, school_idx::Int, route::BirdRoute) = FeasibleRoute(route.stops, sum_individual_travel_times(data, school_idx, route))


mutable struct FeasibleRouteSet
    list::Vector{FeasibleRoute}
    seen::Set{Tuple{Vararg{Int}}}
    at_stop::Vector{Vector{Int}}
end


FeasibleRouteSet(data::BirdData, school_idx::Int) = FeasibleRouteSet(FeasibleRoute[], Set{Tuple{Vararg{Int}}}(), [Int[] for _ in 1:length(data.stops[school_idx])])


function add_route!(routes::FeasibleRouteSet, route::FeasibleRoute)
    key = Tuple(route.stop_ids)
    key in routes.seen && return routes
    push!(routes.seen, key)
    push!(routes.list, route)
    route_idx = length(routes.list)
    for stop_idx in route.stop_ids
        push!(routes.at_stop[stop_idx], route_idx)
    end
    return routes
end


function generate_routes(
    data::BirdData,
    school_idx::Int,
    n_routes::Int,
    max_route_time_lower::Float64,
    max_route_time_upper::Float64;
    rng = Random.default_rng(),
)
    generated = FeasibleRoute[]
    for _ in 1:n_routes
        max_route_time =
            isfinite(max_route_time_lower) ?
            (max_route_time_upper - max_route_time_lower) * rand(rng) + max_route_time_lower :
            Inf
        append!(generated, [FeasibleRoute(data, school_idx, route) for route in greedy_routes(data, school_idx, max_route_time; rng = rng)])
    end
    return generated
end


function best_routes(
    data::BirdData,
    school_idx::Int,
    routes::FeasibleRouteSet,
    lambda_value::Float64;
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
)
    model = _make_model(; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    @variable(model, route_used[1:length(routes.list)], Bin)
    @objective(model, Min, sum(route_used[idx] * (lambda_value + routes.list[idx].cost) for idx in eachindex(routes.list)))
    @constraint(model, [stop_idx in 1:length(data.stops[school_idx])], sum(route_used[idx] for idx in routes.at_stop[stop_idx]) >= 1)
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("best_routes solve failed with $(status)")
    return [idx for idx in eachindex(routes.list) if value(route_used[idx]) >= 0.5]
end


function deletion_cost(data::BirdData, school_idx::Int, route::FeasibleRoute, stop_idx::Int)
    position = findfirst(==(stop_idx), route.stop_ids)
    position === nothing && return -Inf
    length(route.stop_ids) <= 1 && return -Inf
    stops = data.stops[school_idx]
    school = data.schools[school_idx]
    cost = 0.0
    for idx in position:(length(route.stop_ids) - 1)
        cost -= travel_time(data, stops[route.stop_ids[idx]], stops[route.stop_ids[idx + 1]])
        cost -= stop_time(data, stops[route.stop_ids[idx + 1]])
    end
    cost -= travel_time(data, stops[route.stop_ids[end]], school)
    if position == length(route.stop_ids)
        cost -= (
            travel_time(data, stops[route.stop_ids[position - 1]], stops[stop_idx]) +
            travel_time(data, stops[stop_idx], school) +
            stop_time(data, stops[stop_idx]) -
            travel_time(data, stops[route.stop_ids[position - 1]], school)
        ) * (length(route.stop_ids) - 1)
    elseif position > 1
        cost -= (
            travel_time(data, stops[route.stop_ids[position - 1]], stops[stop_idx]) +
            travel_time(data, stops[stop_idx], stops[route.stop_ids[position + 1]]) +
            stop_time(data, stops[stop_idx]) -
            travel_time(data, stops[route.stop_ids[position - 1]], stops[route.stop_ids[position + 1]])
        ) * (position - 1)
    end
    return cost
end


function split_routes(data::BirdData, school_idx::Int, routes::Vector{FeasibleRoute}, stop_idx::Int)
    costs = [deletion_cost(data, school_idx, route, stop_idx) for route in routes]
    selected_route = argmax(costs)
    for route_idx in eachindex(routes)
        route_idx == selected_route && continue
        new_stop_ids = [current_stop for current_stop in routes[route_idx].stop_ids if current_stop != stop_idx]
        routes[route_idx] = FeasibleRoute(new_stop_ids, routes[route_idx].cost + costs[route_idx])
    end
    return routes
end


function build_solution(data::BirdData, school_idx::Int, route_set::FeasibleRouteSet, selected_routes::Vector{Int})
    routes = copy(route_set.list[selected_routes])
    routes_at_stop = [Int[] for _ in 1:length(data.stops[school_idx])]
    for (route_idx, route) in enumerate(routes)
        for stop_idx in route.stop_ids
            push!(routes_at_stop[stop_idx], route_idx)
        end
    end
    for (stop_idx, intersecting_routes) in enumerate(routes_at_stop)
        if length(intersecting_routes) > 1
            new_routes = split_routes(data, school_idx, routes[intersecting_routes], stop_idx)
            for (offset, route_idx) in enumerate(intersecting_routes)
                routes[route_idx] = new_routes[offset]
            end
        end
    end
    return [BirdRoute(idx, route.stop_ids) for (idx, route) in enumerate(routes) if !isempty(route.stop_ids)]
end


function greedy_combined(
    data::BirdData,
    school_idx::Int,
    n_routes::Int,
    max_route_time_lower::Float64,
    max_route_time_upper::Float64,
    lambda_value::Float64;
    rng = Random.default_rng(),
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
)
    routes = FeasibleRouteSet(data, school_idx)
    for route in generate_routes(data, school_idx, n_routes, max_route_time_lower, max_route_time_upper; rng = rng)
        add_route!(routes, route)
    end
    selected = best_routes(data, school_idx, routes, lambda_value; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    return build_solution(data, school_idx, routes, selected)
end


function greedy_combined(
    data::BirdData,
    school_idx::Int,
    start_routes::Vector{BirdRoute},
    n_routes::Int,
    max_route_time_lower::Float64,
    max_route_time_upper::Float64,
    lambda_value::Float64;
    rng = Random.default_rng(),
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
)
    routes = FeasibleRouteSet(data, school_idx)
    for route in generate_routes(data, school_idx, n_routes, max_route_time_lower, max_route_time_upper; rng = rng)
        add_route!(routes, route)
    end
    for route in start_routes
        add_route!(routes, FeasibleRoute(data, school_idx, route))
    end
    selected = best_routes(data, school_idx, routes, lambda_value; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    return build_solution(data, school_idx, routes, selected)
end


function greedy_combined_iterated(
    data::BirdData,
    school_idx::Int,
    params::BirdScenarioParameters;
    rng = Random.default_rng(),
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
)
    routes = greedy_routes(data, school_idx, Inf; rng = rng)
    for _ in 1:params.n_iterations
        routes = greedy_combined(
            data,
            school_idx,
            routes,
            params.n_greedy,
            params.max_route_time_lower,
            params.max_route_time_upper,
            params.lambda_value;
            rng = rng,
            optimizer = optimizer,
            optimizer_attributes = optimizer_attributes,
        )
    end
    return routes
end


function compute_scenarios!(
    data::BirdData,
    params::Vector{BirdScenarioParameters};
    seed::Int = 1,
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
)
    rng = MersenneTwister(seed)
    scenario_list = Tuple{BirdScenario, Vector{BirdRoute}}[]
    for school in data.schools
        for (param_idx, param) in enumerate(params)
            routes = greedy_combined_iterated(
                data,
                school.id,
                param;
                rng = rng,
                optimizer = optimizer,
                optimizer_attributes = optimizer_attributes,
            )
            push!(scenario_list, (BirdScenario(school.id, param_idx, collect(eachindex(routes))), routes))
        end
    end
    load_routing_scenarios!(data, scenario_list)
    return data
end


function load_routing_scenarios!(data::BirdData, scenario_list)
    data.scenarios = [BirdScenario[] for _ in data.schools]
    data.routes = [BirdRoute[] for _ in data.schools]
    ids = vec([(scenario.school, scenario.id) for (scenario, _) in scenario_list])
    for idx in sortperm(ids)
        scenario, routes = scenario_list[idx]
        route_ids = Int[]
        for route in routes
            existing = findfirst(existing_route -> existing_route.stops == route.stops, data.routes[scenario.school])
            if existing === nothing
                push!(data.routes[scenario.school], BirdRoute(length(data.routes[scenario.school]) + 1, route.stops))
                push!(route_ids, length(data.routes[scenario.school]))
            else
                push!(route_ids, existing)
            end
        end
        push!(data.scenarios[scenario.school], BirdScenario(scenario.school, scenario.id, route_ids))
    end
    return data
end
