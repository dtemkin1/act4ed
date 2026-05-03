struct FleetRouteCandidate
    id::Int
    school::Int
    stops::Vector{Int}
    group_id::Int
    grade_id::Int
    passenger_load::Int
    wheelchair_load::Int
    service_time::Float64
    cost::Float64
    compatible_buses::Vector{Int}
end


const FLEET_SCENARIO_EXACT_CANDIDATE_LIMIT = 600


function _aff_sum(terms)
    expr = AffExpr(0.0)
    for term in terms
        expr += term
    end
    return expr
end


function _route_passenger_load(data::BirdData, school_idx::Int, stops::Vector{Int})
    return sum(data.stops[school_idx][stop_idx].n_students for stop_idx in stops)
end


function _route_wheelchair_load(data::BirdData, school_idx::Int, stops::Vector{Int})
    return sum(data.stops[school_idx][stop_idx].n_wheelchair for stop_idx in stops)
end


function _route_group_id(data::BirdData, school_idx::Int, stops::Vector{Int})
    return data.stops[school_idx][stops[1]].group_id
end


function _route_grade_id(data::BirdData, school_idx::Int, stops::Vector{Int})
    return data.stops[school_idx][stops[1]].grade_id
end


function _route_is_homogeneous(data::BirdData, school_idx::Int, stops::Vector{Int})
    isempty(stops) && return false
    group_id = _route_group_id(data, school_idx, stops)
    grade_id = _route_grade_id(data, school_idx, stops)
    return all(
        data.stops[school_idx][stop_idx].group_id == group_id &&
        data.stops[school_idx][stop_idx].grade_id == grade_id
        for stop_idx in stops
    )
end


function _route_compatible_buses(
    data::BirdData,
    school_idx::Int,
    stops::Vector{Int},
    eligible_bus_indices::Vector{Int},
)
    passenger_load = _route_passenger_load(data, school_idx, stops)
    wheelchair_load = _route_wheelchair_load(data, school_idx, stops)
    first_stop = data.stops[school_idx][stops[1]]
    school = data.schools[school_idx]
    compatible = Int[]
    for bus_idx in eligible_bus_indices
        bus = data.fleet[bus_idx]
        depot = data.depots[bus.depot]
        if (
            passenger_load <= bus.capacity &&
            wheelchair_load <= bus.wheelchair_capacity &&
            isfinite(travel_time(data, depot, first_stop)) &&
            isfinite(travel_time(data, school, depot))
        )
            push!(compatible, bus_idx)
        end
    end
    return compatible
end


function _add_fleet_route_candidate!(
    candidates::Vector{FleetRouteCandidate},
    seen::Set{Tuple{Int, Tuple{Vararg{Int}}}},
    data::BirdData,
    school_idx::Int,
    stops::Vector{Int},
    eligible_bus_indices::Vector{Int},
)
    _route_is_homogeneous(data, school_idx, stops) || return candidates
    key = (school_idx, Tuple(stops))
    key in seen && return candidates
    compatible = _route_compatible_buses(data, school_idx, stops, eligible_bus_indices)
    isempty(compatible) && return candidates
    push!(seen, key)
    route = BirdRoute(0, stops)
    push!(
        candidates,
        FleetRouteCandidate(
            length(candidates) + 1,
            school_idx,
            copy(stops),
            _route_group_id(data, school_idx, stops),
            _route_grade_id(data, school_idx, stops),
            _route_passenger_load(data, school_idx, stops),
            _route_wheelchair_load(data, school_idx, stops),
            service_time(data, school_idx, route),
            sum_individual_travel_times(data, school_idx, route),
            compatible,
        ),
    )
    return candidates
end


function _restricted_greedy_routes(
    data::BirdData,
    school_idx::Int,
    allowed::BitVector,
    max_route_time::Float64,
    passenger_capacity::Int,
    wheelchair_capacity::Int;
    rng = Random.default_rng(),
)
    routes = BirdRoute[]
    available = copy(allowed)
    while any(available)
        start_options = findall(identity, available)
        start_stop_idx = rand(rng, start_options)
        state = initial_route(data, school_idx, start_stop_idx, length(routes) + 1)
        current_wheelchair = data.stops[school_idx][start_stop_idx].n_wheelchair
        available[start_stop_idx] = false
        while true
            best_stop_idx = 0
            best_insert_idx = -1
            best_time_diff = Inf
            for stop_idx in findall(identity, available)
                stop = data.stops[school_idx][stop_idx]
                if (
                    stop.group_id == data.stops[school_idx][start_stop_idx].group_id &&
                    stop.grade_id == state.grade_id &&
                    stop.n_students + state.n_students <= passenger_capacity &&
                    stop.n_wheelchair + current_wheelchair <= wheelchair_capacity
                )
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
                current_wheelchair += data.stops[school_idx][best_stop_idx].n_wheelchair
                available[best_stop_idx] = false
            else
                push!(routes, state.route)
                break
            end
        end
    end
    return routes
end


function _sampled_max_route_time(params::BirdScenarioParameters, rng)
    if isfinite(params.max_route_time_lower)
        return (params.max_route_time_upper - params.max_route_time_lower) * rand(rng) + params.max_route_time_lower
    end
    return Inf
end


function _generate_fleet_route_candidates(
    data::BirdData,
    available::Vector{BitVector},
    eligible_bus_indices::Vector{Int},
    scenario_params::Vector{BirdScenarioParameters};
    rng = Random.default_rng(),
)
    candidates = FleetRouteCandidate[]
    seen = Set{Tuple{Int, Tuple{Vararg{Int}}}}()
    isempty(eligible_bus_indices) && return candidates
    passenger_capacity = maximum(data.fleet[bus_idx].capacity for bus_idx in eligible_bus_indices)
    wheelchair_capacity = maximum(data.fleet[bus_idx].wheelchair_capacity for bus_idx in eligible_bus_indices)

    for school_idx in eachindex(available)
        for stop_idx in findall(identity, available[school_idx])
            stop = data.stops[school_idx][stop_idx]
            if (
                stop.n_students <= passenger_capacity &&
                stop.n_wheelchair <= wheelchair_capacity &&
                travel_time(data, stop, data.schools[school_idx]) <= max_travel_time(data, stop)
            )
                _add_fleet_route_candidate!(candidates, seen, data, school_idx, [stop_idx], eligible_bus_indices)
            end
        end

        grade_ids = unique(data.stops[school_idx][stop_idx].grade_id for stop_idx in findall(identity, available[school_idx]))
        for grade_id in grade_ids
            allowed = BitVector([
                available[school_idx][stop_idx] &&
                data.stops[school_idx][stop_idx].grade_id == grade_id
                for stop_idx in eachindex(data.stops[school_idx])
            ])
            any(allowed) || continue
            for params in scenario_params
                for _ in 1:(params.n_greedy * max(params.n_iterations, 1))
                    max_route_time = _sampled_max_route_time(params, rng)
                    for route in _restricted_greedy_routes(
                        data,
                        school_idx,
                        allowed,
                        max_route_time,
                        passenger_capacity,
                        wheelchair_capacity;
                        rng = rng,
                    )
                        _add_fleet_route_candidate!(
                            candidates,
                            seen,
                            data,
                            school_idx,
                            route.stops,
                            eligible_bus_indices,
                        )
                    end
                end
            end
        end
    end
    return candidates
end


function _candidate_route_to_route_feasible(data::BirdData, first::FleetRouteCandidate, second::FleetRouteCandidate)
    first.school == second.school && return false
    route = BirdRoute(0, second.stops)
    return is_feasible_in_time(data, first.school, second.school, route, second.service_time)
end


function _candidate_route_arcs(data::BirdData, candidates::Vector{FleetRouteCandidate})
    arcs = Tuple{Int, Int}[]
    for first_idx in eachindex(candidates), second_idx in eachindex(candidates)
        first_idx == second_idx && continue
        if _candidate_route_to_route_feasible(data, candidates[first_idx], candidates[second_idx])
            push!(arcs, (first_idx, second_idx))
        end
    end
    return arcs
end


function _stage_stop_pairs(available::Vector{BitVector})
    return [
        (school_idx, stop_idx)
        for school_idx in eachindex(available)
        for stop_idx in findall(identity, available[school_idx])
    ]
end


function _coverage_by_stop(candidates::Vector{FleetRouteCandidate}, stage_stops::Vector{Tuple{Int, Int}})
    stop_to_position = Dict(stop => idx for (idx, stop) in enumerate(stage_stops))
    covering = [Int[] for _ in stage_stops]
    for route_idx in eachindex(candidates)
        candidate = candidates[route_idx]
        for stop_idx in candidate.stops
            position = get(stop_to_position, (candidate.school, stop_idx), nothing)
            position === nothing || push!(covering[position], route_idx)
        end
    end
    return covering
end


function _reduce_fleet_route_candidates(
    data::BirdData,
    candidates::Vector{FleetRouteCandidate},
    stage_stops::Vector{Tuple{Int, Int}},
    label::AbstractString;
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
)
    length(candidates) <= FLEET_SCENARIO_EXACT_CANDIDATE_LIMIT && return candidates

    stage_keys = Set{Tuple{Int, Int, Int}}()
    for (school_idx, stop_idx) in stage_stops
        stop = data.stops[school_idx][stop_idx]
        push!(stage_keys, (school_idx, stop.group_id, stop.grade_id))
    end

    selected_indices = Set{Int}()
    for key in sort(collect(stage_keys))
        school_idx, group_id, grade_id = key
        group_candidate_indices = [
            idx for idx in eachindex(candidates)
            if candidates[idx].school == school_idx &&
               candidates[idx].group_id == group_id &&
               candidates[idx].grade_id == grade_id
        ]
        isempty(group_candidate_indices) && continue

        group_stops = [
            stop_idx for (candidate_school, stop_idx) in stage_stops
            if candidate_school == school_idx &&
               data.stops[school_idx][stop_idx].group_id == group_id &&
               data.stops[school_idx][stop_idx].grade_id == grade_id
        ]
        covering = Dict{Int, Vector{Int}}(stop_idx => Int[] for stop_idx in group_stops)
        for idx in group_candidate_indices
            for stop_idx in candidates[idx].stops
                if haskey(covering, stop_idx)
                    push!(covering[stop_idx], idx)
                end
            end
        end
        coverable_stops = [stop_idx for stop_idx in group_stops if !isempty(covering[stop_idx])]
        isempty(coverable_stops) && continue

        model = _make_model(; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
        @variable(model, pick[group_candidate_indices], Bin)
        for stop_idx in coverable_stops
            @constraint(model, _aff_sum(pick[idx] for idx in covering[stop_idx]) == 1)
        end
        @objective(
            model,
            Min,
            _aff_sum(
                pick[idx] * (data.default_lambda_value + candidates[idx].cost)
                for idx in group_candidate_indices
            ),
        )
        optimize!(model)
        status = termination_status(model)
        status == MOI.OPTIMAL || error("fleet-aware scenario $(label) candidate reduction failed with $(status)")
        for idx in group_candidate_indices
            value(pick[idx]) >= 0.5 && push!(selected_indices, idx)
        end
    end

    return [candidates[idx] for idx in eachindex(candidates) if idx in selected_indices]
end


function _route_start_cost(data::BirdData, bus_idx::Int, candidate::FleetRouteCandidate)
    depot = data.depots[data.fleet[bus_idx].depot]
    first_stop = data.stops[candidate.school][candidate.stops[1]]
    return travel_time(data, depot, first_stop)
end


function _route_finish_cost(data::BirdData, bus_idx::Int, candidate::FleetRouteCandidate)
    school = data.schools[candidate.school]
    depot = data.depots[data.fleet[bus_idx].depot]
    return travel_time(data, school, depot)
end


function _route_link_cost(data::BirdData, first::FleetRouteCandidate, second::FleetRouteCandidate)
    school = data.schools[first.school]
    first_stop = data.stops[second.school][second.stops[1]]
    return travel_time(data, school, first_stop)
end


function _solve_fleet_stage_mip!(
    data::BirdData,
    routes::Vector{Vector{BirdRoute}},
    buses::Vector{BirdBus},
    available::Vector{BitVector},
    remaining_bus_indices::Set{Int},
    eligible_bus_indices::Vector{Int},
    label::AbstractString,
    scenario_params::Vector{BirdScenarioParameters};
    rng = Random.default_rng(),
    fail_if_unserved::Bool = true,
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
    timing_log::Bool = false,
)
    stage_stops = _stage_stop_pairs(available)
    isempty(stage_stops) && return Int[]

    eligible_remaining = [bus_idx for bus_idx in eligible_bus_indices if bus_idx in remaining_bus_indices]
    if isempty(eligible_remaining)
        fail_if_unserved && error("insufficient fleet for $(label) Bird demand: no eligible buses remain")
        return Int[]
    end

    candidates = _timed_value("fleet scenario $(label) candidate generation", timing_log) do
        _generate_fleet_route_candidates(
            data,
            available,
            eligible_remaining,
            scenario_params;
            rng = rng,
        )
    end
    candidates = _timed_value("fleet scenario $(label) candidate reduction", timing_log) do
        _reduce_fleet_route_candidates(
            data,
            candidates,
            stage_stops,
            label;
            optimizer = optimizer,
            optimizer_attributes = optimizer_attributes,
        )
    end
    covering = _timed_value("fleet scenario $(label) coverage indexing", timing_log) do
        _coverage_by_stop(candidates, stage_stops)
    end
    if fail_if_unserved
        for (position, routes_covering_stop) in enumerate(covering)
            if isempty(routes_covering_stop)
                school_idx, stop_idx = stage_stops[position]
                stop = data.stops[school_idx][stop_idx]
                error("insufficient fleet for $(label) Bird demand: no feasible route candidate covers stop $(stop.external_id)")
            end
        end
    end
    isempty(candidates) && return Int[]

    bus_ids = collect(eligible_remaining)
    route_ids = collect(eachindex(candidates))
    route_arcs = _timed_value("fleet scenario $(label) arc construction", timing_log) do
        _candidate_route_arcs(data, candidates)
    end
    compatible_bus_sets = [Set(candidate.compatible_buses) for candidate in candidates]
    _log_timing_message(
        "fleet scenario $(label) size: stops=$(length(stage_stops)), buses=$(length(bus_ids)), candidates=$(length(route_ids)), arcs=$(length(route_arcs)), assign_vars=$(length(bus_ids) * length(route_ids)), link_vars=$(length(bus_ids) * length(route_arcs))",
        timing_log,
    )

    model_build_start = time()
    model = _make_model(; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    @variable(model, assign[bus_ids, route_ids], Bin)
    @variable(model, used[bus_ids], Bin)
    @variable(model, start_route[bus_ids, route_ids], Bin)
    @variable(model, finish_route[bus_ids, route_ids], Bin)
    @variable(model, link_route[bus_ids, route_arcs], Bin)
    @variable(model, served[1:length(stage_stops)], Bin)

    for bus_idx in bus_ids, route_idx in route_ids
        if !(bus_idx in compatible_bus_sets[route_idx])
            @constraint(model, assign[bus_idx, route_idx] == 0)
            @constraint(model, start_route[bus_idx, route_idx] == 0)
            @constraint(model, finish_route[bus_idx, route_idx] == 0)
        end
    end

    for bus_idx in bus_ids, (first_idx, second_idx) in route_arcs
        if !(bus_idx in compatible_bus_sets[first_idx]) || !(bus_idx in compatible_bus_sets[second_idx])
            @constraint(model, link_route[bus_idx, (first_idx, second_idx)] == 0)
        end
    end

    for route_idx in route_ids
        @constraint(model, _aff_sum(assign[bus_idx, route_idx] for bus_idx in bus_ids) <= 1)
    end

    for bus_idx in bus_ids
        @constraint(model, _aff_sum(start_route[bus_idx, route_idx] for route_idx in route_ids) == used[bus_idx])
        @constraint(model, _aff_sum(finish_route[bus_idx, route_idx] for route_idx in route_ids) == used[bus_idx])
        for route_idx in route_ids
            incoming = _aff_sum(link_route[bus_idx, arc] for arc in route_arcs if arc[2] == route_idx)
            outgoing = _aff_sum(link_route[bus_idx, arc] for arc in route_arcs if arc[1] == route_idx)
            @constraint(model, assign[bus_idx, route_idx] == start_route[bus_idx, route_idx] + incoming)
            @constraint(model, assign[bus_idx, route_idx] == finish_route[bus_idx, route_idx] + outgoing)
            @constraint(model, assign[bus_idx, route_idx] <= used[bus_idx])
        end
        for school_idx in eachindex(data.schools)
            @constraint(
                model,
                _aff_sum(
                    assign[bus_idx, route_idx]
                    for route_idx in route_ids
                    if candidates[route_idx].school == school_idx
                ) <= 1,
            )
        end
    end

    for position in eachindex(stage_stops)
        coverage = _aff_sum(
            assign[bus_idx, route_idx]
            for bus_idx in bus_ids
            for route_idx in covering[position]
        )
        if data.allow_partial
            @constraint(model, coverage == served[position])
        else
            @constraint(model, coverage == 1)
            @constraint(model, served[position] == 1)
        end
    end

    served_objective = _aff_sum(
        data.stops[stage_stops[position][1]][stage_stops[position][2]].n_students * served[position]
        for position in eachindex(stage_stops)
    )
    if data.allow_partial
        @objective(model, Max, served_objective)
        _log_timing("fleet scenario $(label) model build", timing_log, time() - model_build_start)
        _timed_value("fleet scenario $(label) served optimize", timing_log) do
            optimize!(model)
        end
        status = termination_status(model)
        status == MOI.OPTIMAL || error("fleet-aware scenario $(label) served-count solve failed with $(status)")
        best_served = objective_value(model)
        @constraint(model, served_objective >= best_served - BIRD_TIMING_EPS)
    else
        _log_timing("fleet scenario $(label) model build", timing_log, time() - model_build_start)
    end

    bus_fixed_cost = max(data.default_lambda_value * 100.0, 1.0e6)
    objective_build_start = time()
    cost_objective = _timed_value("fleet scenario $(label) objective bus fixed term", timing_log) do
        bus_fixed_cost * _aff_sum(used[bus_idx] for bus_idx in bus_ids)
    end
    cost_objective += _timed_value("fleet scenario $(label) objective route assignment term", timing_log) do
        _aff_sum(
            assign[bus_idx, route_idx] * candidates[route_idx].cost
            for bus_idx in bus_ids
            for route_idx in route_ids
        )
    end
    cost_objective += _timed_value("fleet scenario $(label) objective route start term", timing_log) do
        _aff_sum(
            start_route[bus_idx, route_idx] * _route_start_cost(data, bus_idx, candidates[route_idx])
            for bus_idx in bus_ids
            for route_idx in route_ids
        )
    end
    cost_objective += _timed_value("fleet scenario $(label) objective route finish term", timing_log) do
        _aff_sum(
            finish_route[bus_idx, route_idx] * _route_finish_cost(data, bus_idx, candidates[route_idx])
            for bus_idx in bus_ids
            for route_idx in route_ids
        )
    end
    cost_objective += _timed_value("fleet scenario $(label) objective route link term", timing_log) do
        _aff_sum(
            link_route[bus_idx, arc] * _route_link_cost(data, candidates[arc[1]], candidates[arc[2]])
            for bus_idx in bus_ids
            for arc in route_arcs
        )
    end
    _timed_value("fleet scenario $(label) objective attach", timing_log) do
        @objective(model, Min, cost_objective)
    end
    _log_timing("fleet scenario $(label) objective build", timing_log, time() - objective_build_start)
    _timed_value("fleet scenario $(label) cost optimize", timing_log) do
        optimize!(model)
    end
    status = termination_status(model)
    if status != MOI.OPTIMAL
        fail_if_unserved && error("fleet-aware scenario $(label) solve failed with $(status)")
        return Int[]
    end

    used_buses = Int[]
    for bus_idx in bus_ids
        value(used[bus_idx]) >= 0.5 || continue
        start_candidates = [route_idx for route_idx in route_ids if value(start_route[bus_idx, route_idx]) >= 0.5]
        length(start_candidates) == 1 || error("fleet-aware scenario $(label) produced inconsistent start route for bus $(bus_idx)")
        route_path = Int[]
        current_route = start_candidates[1]
        while true
            push!(route_path, current_route)
            next_routes = [arc[2] for arc in route_arcs if arc[1] == current_route && value(link_route[bus_idx, arc]) >= 0.5]
            if isempty(next_routes)
                break
            end
            length(next_routes) == 1 || error("fleet-aware scenario $(label) produced branching route path for bus $(bus_idx)")
            current_route = next_routes[1]
            current_route in route_path && error("fleet-aware scenario $(label) produced a route cycle for bus $(bus_idx)")
        end

        assigned_routes = Set(route_idx for route_idx in route_ids if value(assign[bus_idx, route_idx]) >= 0.5)
        Set(route_path) == assigned_routes || error("fleet-aware scenario $(label) produced disconnected route assignments for bus $(bus_idx)")

        bus_schools = Int[]
        bus_routes = Int[]
        first_stops = Int[]
        route_times = Float64[]
        for route_idx in route_path
            candidate = candidates[route_idx]
            push!(routes[candidate.school], BirdRoute(length(routes[candidate.school]) + 1, candidate.stops))
            push!(bus_schools, candidate.school)
            push!(bus_routes, length(routes[candidate.school]))
            push!(first_stops, candidate.stops[1])
            push!(route_times, candidate.service_time)
            for stop_idx in candidate.stops
                available[candidate.school][stop_idx] = false
            end
        end
        arrival_times = school_route_arrival_times(data, bus_schools, first_stops, route_times)
        arrival_times === nothing && error("fleet-aware scenario $(label) produced an infeasible arrival schedule")
        push!(buses, BirdBus(bus_idx, data.fleet[bus_idx].depot, bus_schools, bus_routes, arrival_times))
        push!(used_buses, bus_idx)
    end
    return used_buses
end


function solve_fleet_aware_with_scenarios!(
    data::BirdData;
    scenario_params = default_scenario_parameters(data),
    seed::Int = 1,
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
    timing_log::Bool = false,
)
    isempty(data.fleet) && error("fleet-aware Bird instance has no concrete buses")

    rng = MersenneTwister(seed)
    routes = [BirdRoute[] for _ in data.schools]
    buses = BirdBus[]
    remaining_bus_indices = Set(eachindex(data.fleet))

    wheelchair_available = available_for_group(data, SERVICE_GROUP_WHEELCHAIR)
    wheelchair_buses = [
        idx for idx in eachindex(data.fleet)
        if data.fleet[idx].has_monitor && data.fleet[idx].wheelchair_capacity > 0
    ]
    used = _solve_fleet_stage_mip!(
        data,
        routes,
        buses,
        wheelchair_available,
        remaining_bus_indices,
        wheelchair_buses,
        "wheelchair",
        scenario_params;
        rng = rng,
        fail_if_unserved = !data.allow_partial,
        optimizer = optimizer,
        optimizer_attributes = optimizer_attributes,
        timing_log = timing_log,
    )
    foreach(bus_idx -> delete!(remaining_bus_indices, bus_idx), used)

    sped_available = available_for_group(data, SERVICE_GROUP_SPED)
    sped_buses = [idx for idx in eachindex(data.fleet) if data.fleet[idx].has_monitor]
    used = _solve_fleet_stage_mip!(
        data,
        routes,
        buses,
        sped_available,
        remaining_bus_indices,
        sped_buses,
        "SPED",
        scenario_params;
        rng = rng,
        fail_if_unserved = !data.allow_partial,
        optimizer = optimizer,
        optimizer_attributes = optimizer_attributes,
        timing_log = timing_log,
    )
    foreach(bus_idx -> delete!(remaining_bus_indices, bus_idx), used)

    conventional_available = available_for_group(data, SERVICE_GROUP_CONVENTIONAL)
    non_monitor_buses = [idx for idx in eachindex(data.fleet) if !data.fleet[idx].has_monitor]
    used = _solve_fleet_stage_mip!(
        data,
        routes,
        buses,
        conventional_available,
        remaining_bus_indices,
        non_monitor_buses,
        "conventional non-monitor",
        scenario_params;
        rng = rng,
        fail_if_unserved = !data.allow_partial && !data.conventional_spillover,
        optimizer = optimizer,
        optimizer_attributes = optimizer_attributes,
        timing_log = timing_log,
    )
    foreach(bus_idx -> delete!(remaining_bus_indices, bus_idx), used)

    if any_available(conventional_available)
        if !data.conventional_spillover && !data.allow_partial
            error("insufficient fleet for conventional Bird demand: non-monitor buses were exhausted before all stops were routed")
        end
        if data.conventional_spillover
            used = _solve_fleet_stage_mip!(
                data,
                routes,
                buses,
                conventional_available,
                remaining_bus_indices,
                collect(eachindex(data.fleet)),
                "conventional spillover",
                scenario_params;
                rng = rng,
                fail_if_unserved = !data.allow_partial,
                optimizer = optimizer,
                optimizer_attributes = optimizer_attributes,
                timing_log = timing_log,
            )
            foreach(bus_idx -> delete!(remaining_bus_indices, bus_idx), used)
        end
    end

    data.routes = routes
    data.scenarios = [[BirdScenario(idx, 1, collect(eachindex(routes[idx])))] for idx in eachindex(data.schools)]
    data.used_scenario = ones(Int, length(data.schools))
    data.buses = buses
    data.unassigned_stops = vcat(
        available_stop_pairs(wheelchair_available),
        available_stop_pairs(sped_available),
        available_stop_pairs(conventional_available),
    )
    return data
end
