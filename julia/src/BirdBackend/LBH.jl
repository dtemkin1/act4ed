struct Itinerary
    schools::Vector{Int}
    stops::Vector{Vector{Int}}
    n_students::Vector{Int}
    n_wheelchair::Vector{Int}
    slack_times::Vector{Vector{Float64}}
    stop_times::Vector{Vector{Float64}}
    route_times::Vector{Float64}
end


function random_stop(available_stops::Vector{BitVector}; rng = Random.default_rng())
    candidates = Tuple{Int, Int}[]
    for school_idx in eachindex(available_stops), stop_idx in eachindex(available_stops[school_idx])
        available_stops[school_idx][stop_idx] && push!(candidates, (school_idx, stop_idx))
    end
    return rand(rng, candidates)
end


function under_capacity(
    data::BirdData,
    itinerary::Itinerary,
    school_idx::Int,
    stop_idx::Int,
    passenger_capacity::Int,
    wheelchair_capacity::Int,
)
    current_idx = findfirst(==(school_idx), itinerary.schools)
    current_idx === nothing && return true
    stop = data.stops[school_idx][stop_idx]
    return (
        stop.n_students + itinerary.n_students[current_idx] <= passenger_capacity &&
        stop.n_wheelchair + itinerary.n_wheelchair[current_idx] <= wheelchair_capacity
    )
end


under_capacity(data::BirdData, itinerary::Itinerary, school_idx::Int, stop_idx::Int) =
    under_capacity(data, itinerary, school_idx, stop_idx, data.params.bus_capacity, typemax(Int))


any_available(available_stops::Vector{BitVector}) = any(any(mask) for mask in available_stops)


function available_stop_pairs(available_stops::Vector{BitVector})
    return [
        (school_idx, stop_idx)
        for school_idx in eachindex(available_stops)
        for stop_idx in findall(identity, available_stops[school_idx])
    ]
end


function stop_fits_bus(data::BirdData, school_idx::Int, stop_idx::Int, bus::BirdFleetBus)
    stop = data.stops[school_idx][stop_idx]
    return (
        stop.n_students <= bus.capacity &&
        stop.n_wheelchair <= bus.wheelchair_capacity &&
        travel_time(data, stop, data.schools[school_idx]) <= max_travel_time(data, stop)
    )
end


function itinerary_grade_id(data::BirdData, itinerary::Itinerary)
    return data.stops[itinerary.schools[1]][itinerary.stops[1][1]].grade_id
end


function stop_matches_itinerary_grade(data::BirdData, itinerary::Itinerary, school_idx::Int, stop_idx::Int)
    return data.stops[school_idx][stop_idx].grade_id == itinerary_grade_id(data, itinerary)
end


function available_for_group(data::BirdData, group_id::Int)
    return [
        BitVector([stop.group_id == group_id for stop in data.stops[school_idx]])
        for school_idx in eachindex(data.schools)
    ]
end


function ordered_bus_indices(data::BirdData, indices::Vector{Int}; prefer_non_monitor::Bool = false)
    return sort(
        indices;
        by = idx -> (
            prefer_non_monitor && data.fleet[idx].has_monitor ? 1 : 0,
            data.fleet[idx].wheelchair_capacity,
            data.fleet[idx].capacity,
            idx,
        ),
    )
end


function assert_available_stops_fit_buses(data::BirdData, available::Vector{BitVector}, eligible_bus_indices::Vector{Int}, label::AbstractString)
    any_available(available) || return true
    isempty(eligible_bus_indices) && error("insufficient fleet for $(label) Bird demand: no eligible buses remain")
    for school_idx in eachindex(available)
        for stop_idx in findall(identity, available[school_idx])
            if !any(stop_fits_bus(data, school_idx, stop_idx, data.fleet[bus_idx]) for bus_idx in eligible_bus_indices)
                stop = data.stops[school_idx][stop_idx]
                error(
                    "insufficient fleet for $(label) Bird demand: stop $(stop.external_id) has $(stop.n_students) students and $(stop.n_wheelchair) wheelchair students, which exceeds every eligible remaining bus",
                )
            end
        end
    end
    return true
end


function build_itinerary_for_bus!(
    data::BirdData,
    routes::Vector{Vector{BirdRoute}},
    available::Vector{BitVector},
    bus::BirdFleetBus;
    rng = Random.default_rng(),
)
    candidates = Tuple{Int, Int}[]
    for school_idx in eachindex(available), stop_idx in findall(identity, available[school_idx])
        stop_fits_bus(data, school_idx, stop_idx, bus) && push!(candidates, (school_idx, stop_idx))
    end
    isempty(candidates) && return nothing

    school_idx, stop_idx = rand(rng, candidates)
    itinerary = initial_itinerary(data, school_idx, stop_idx)
    available[school_idx][stop_idx] = false
    use_original_timing = uses_original_dwell_timing(data)
    while true
        best_school = 0
        best_stop = 0
        best_insert = (-1, -1)
        best_cost = Inf
        for candidate_school in eachindex(available)
            for candidate_stop in findall(identity, available[candidate_school])
                stop_fits_bus(data, candidate_school, candidate_stop, bus) || continue
                stop_matches_itinerary_grade(data, itinerary, candidate_school, candidate_stop) || continue
                under_capacity(data, itinerary, candidate_school, candidate_stop, bus.capacity, bus.wheelchair_capacity) || continue
                insert_school, insert_stop, cost =
                    use_original_timing ?
                    itinerary_best_insertion_original(data, candidate_school, candidate_stop, itinerary) :
                    itinerary_best_insertion_window(data, candidate_school, candidate_stop, itinerary)
                if cost < best_cost
                    best_cost = cost
                    best_school = candidate_school
                    best_stop = candidate_stop
                    best_insert = (insert_school, insert_stop)
                end
            end
        end
        if isfinite(best_cost)
            itinerary = insert_itinerary(data, itinerary, best_school, best_stop, best_insert[1], best_insert[2])
            available[best_school][best_stop] = false
        else
            bus_schools = Int[]
            bus_routes = Int[]
            for (offset, school) in enumerate(itinerary.schools)
                push!(routes[school], BirdRoute(length(routes[school]) + 1, itinerary.stops[offset]))
                push!(bus_schools, school)
                push!(bus_routes, length(routes[school]))
            end
            arrival_times = itinerary_arrival_times(data, itinerary)
            arrival_times === nothing && error("internal Bird LBH timing failure")
            return BirdBus(bus.id, bus.depot, bus_schools, bus_routes, arrival_times)
        end
    end
end


function route_group_with_buses!(
    data::BirdData,
    routes::Vector{Vector{BirdRoute}},
    buses::Vector{BirdBus},
    available::Vector{BitVector},
    remaining_bus_indices::Set{Int},
    eligible_bus_indices::Vector{Int},
    label::AbstractString;
    rng = Random.default_rng(),
    fail_if_unserved::Bool = true,
    prefer_non_monitor::Bool = false,
)
    eligible_remaining = [idx for idx in eligible_bus_indices if idx in remaining_bus_indices]
    if fail_if_unserved
        assert_available_stops_fit_buses(data, available, eligible_remaining, label)
    end

    for bus_idx in ordered_bus_indices(data, eligible_remaining; prefer_non_monitor = prefer_non_monitor)
        any_available(available) || break
        bus = data.fleet[bus_idx]
        used_bus = build_itinerary_for_bus!(data, routes, available, bus; rng = rng)
        if used_bus !== nothing
            push!(buses, used_bus)
            delete!(remaining_bus_indices, bus_idx)
        end
    end

    if fail_if_unserved && any_available(available)
        error("insufficient fleet for $(label) Bird demand: eligible buses were exhausted before all stops were routed")
    end
    return available
end


function solve_fleet_aware!(data::BirdData; seed::Int = 1)
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
    route_group_with_buses!(
        data,
        routes,
        buses,
        wheelchair_available,
        remaining_bus_indices,
        wheelchair_buses,
        "wheelchair";
        rng = rng,
        fail_if_unserved = !data.allow_partial,
    )

    sped_available = available_for_group(data, SERVICE_GROUP_SPED)
    sped_buses = [idx for idx in eachindex(data.fleet) if data.fleet[idx].has_monitor]
    route_group_with_buses!(
        data,
        routes,
        buses,
        sped_available,
        remaining_bus_indices,
        sped_buses,
        "SPED";
        rng = rng,
        fail_if_unserved = !data.allow_partial,
    )

    conventional_available = available_for_group(data, SERVICE_GROUP_CONVENTIONAL)
    non_monitor_buses = [idx for idx in eachindex(data.fleet) if !data.fleet[idx].has_monitor]
    route_group_with_buses!(
        data,
        routes,
        buses,
        conventional_available,
        remaining_bus_indices,
        non_monitor_buses,
        "conventional non-monitor";
        rng = rng,
        fail_if_unserved = !data.allow_partial && !data.conventional_spillover,
        prefer_non_monitor = true,
    )
    if any_available(conventional_available)
        if !data.conventional_spillover && !data.allow_partial
            error("insufficient fleet for conventional Bird demand: non-monitor buses were exhausted before all stops were routed")
        end
        if data.conventional_spillover
            all_buses = collect(eachindex(data.fleet))
            route_group_with_buses!(
                data,
                routes,
                buses,
                conventional_available,
                remaining_bus_indices,
                all_buses,
                "conventional spillover";
                rng = rng,
                fail_if_unserved = !data.allow_partial,
                prefer_non_monitor = true,
            )
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


function initial_itinerary(data::BirdData, school_idx::Int, stop_idx::Int)
    stop = data.stops[school_idx][stop_idx]
    school = data.schools[school_idx]
    time_on_bus = travel_time(data, stop, school)
    return Itinerary(
        [school_idx],
        [[stop_idx]],
        [stop.n_students],
        [stop.n_wheelchair],
        [[max_travel_time(data, stop) - time_on_bus]],
        [[time_on_bus]],
        [time_on_bus + stop_time(data, stop)],
    )
end


function itinerary_first_stops(itinerary::Itinerary)
    return [route_stops[1] for route_stops in itinerary.stops]
end


function itinerary_arrival_times(data::BirdData, itinerary::Itinerary)
    return school_route_arrival_times(
        data,
        itinerary.schools,
        itinerary_first_stops(itinerary),
        itinerary.route_times,
    )
end


function itinerary_respects_ride_times(itinerary::Itinerary)
    return all(all(slack .>= -BIRD_TIMING_EPS) for slack in itinerary.slack_times)
end


function itinerary_chain_time(data::BirdData, itinerary::Itinerary)
    total = sum(itinerary.route_times)
    for idx in 2:length(itinerary.schools)
        total += travel_time(
            data,
            data.schools[itinerary.schools[idx - 1]],
            data.stops[itinerary.schools[idx]][itinerary.stops[idx][1]],
        )
    end
    return total
end


function itinerary_feasible(data::BirdData, itinerary::Itinerary)
    itinerary_respects_ride_times(itinerary) || return false
    itinerary_arrival_times(data, itinerary) !== nothing || return false
    return true
end


function itinerary_best_insertion_original(data::BirdData, school_idx::Int, new_stop_idx::Int, itinerary::Itinerary)
    school_pos = findfirst(==(school_idx), itinerary.schools)
    new_stop = data.stops[school_idx][new_stop_idx]
    if school_pos === nothing
        best_time_diff = Inf
        insert_school = -1
        if data.schools[school_idx].start_time + travel_time(data, data.schools[school_idx], data.stops[itinerary.schools[1]][itinerary.stops[1][1]]) + itinerary.route_times[1] + data.schools[itinerary.schools[1]].dwell_time <= data.schools[itinerary.schools[1]].start_time
            best_time_diff =
                stop_time(data, new_stop) +
                travel_time(data, new_stop, data.schools[school_idx]) +
                data.schools[school_idx].dwell_time +
                travel_time(data, data.schools[school_idx], data.stops[itinerary.schools[1]][itinerary.stops[1][1]])
            insert_school = 0
        end
        for idx in 1:(length(itinerary.schools) - 1)
            feasible =
                data.schools[itinerary.schools[idx]].start_time +
                travel_time(data, data.schools[itinerary.schools[idx]], new_stop) +
                stop_time(data, new_stop) +
                travel_time(data, new_stop, data.schools[school_idx]) +
                data.schools[school_idx].dwell_time <= data.schools[school_idx].start_time &&
                data.schools[school_idx].start_time +
                travel_time(data, data.schools[school_idx], data.stops[itinerary.schools[idx + 1]][itinerary.stops[idx + 1][1]]) +
                itinerary.route_times[idx + 1] +
                data.schools[itinerary.schools[idx + 1]].dwell_time <= data.schools[itinerary.schools[idx + 1]].start_time
            if feasible
                time_diff =
                    travel_time(data, data.schools[itinerary.schools[idx]], new_stop) +
                    stop_time(data, new_stop) +
                    travel_time(data, new_stop, data.schools[school_idx]) +
                    data.schools[school_idx].dwell_time +
                    travel_time(data, data.schools[school_idx], data.stops[itinerary.schools[idx + 1]][itinerary.stops[idx + 1][1]])
                if time_diff < best_time_diff
                    best_time_diff = time_diff
                    insert_school = idx
                end
            end
        end
        time_diff =
            travel_time(data, data.schools[itinerary.schools[end]], new_stop) +
            stop_time(data, new_stop) +
            travel_time(data, new_stop, data.schools[school_idx]) +
            data.schools[school_idx].dwell_time
        if data.schools[itinerary.schools[end]].start_time + time_diff <= data.schools[school_idx].start_time && time_diff < best_time_diff
            best_time_diff = time_diff
            insert_school = length(itinerary.schools)
        end
        return insert_school, -1, best_time_diff
    end

    school = data.schools[school_idx]
    current_stops = itinerary.stops[school_pos]
    best_time_diff = Inf
    insert_stop = -1
    time_to_next = travel_time(data, new_stop, data.stops[school_idx][current_stops[1]])
    time_diff =
        school_pos > 1 ?
        travel_time(data, data.schools[itinerary.schools[school_pos - 1]], new_stop) +
        time_to_next +
        stop_time(data, new_stop) -
        travel_time(data, data.schools[itinerary.schools[school_pos - 1]], data.stops[school_idx][current_stops[1]]) :
        time_to_next + stop_time(data, new_stop)
    feasible =
        (school_pos == 1 || data.schools[itinerary.schools[school_pos - 1]].start_time + itinerary.route_times[school_pos] + time_diff + school.dwell_time <= school.start_time) &&
        time_to_next + itinerary.route_times[school_pos] <= max_travel_time(data, new_stop)
    if feasible
        best_time_diff = time_diff
        insert_stop = 0
    end

    for idx in 1:(length(current_stops) - 1)
        time_to_next = travel_time(data, new_stop, data.stops[school_idx][current_stops[idx + 1]])
        time_diff =
            travel_time(data, data.stops[school_idx][current_stops[idx]], new_stop) +
            time_to_next -
            travel_time(data, data.stops[school_idx][current_stops[idx]], data.stops[school_idx][current_stops[idx + 1]]) +
            stop_time(data, new_stop)
        feasible =
            time_diff <= itinerary.slack_times[school_pos][idx] &&
            itinerary.stop_times[school_pos][idx + 1] + time_to_next + stop_time(data, data.stops[school_idx][current_stops[idx + 1]]) <= max_travel_time(data, new_stop) &&
            (school_pos == 1 || data.schools[itinerary.schools[school_pos - 1]].start_time + itinerary.route_times[school_pos] + time_diff + school.dwell_time <= school.start_time)
        if feasible && time_diff < best_time_diff
            best_time_diff = time_diff
            insert_stop = idx
        end
    end

    time_diff =
        travel_time(data, data.stops[school_idx][current_stops[end]], new_stop) +
        travel_time(data, new_stop, school) +
        stop_time(data, new_stop) -
        travel_time(data, data.stops[school_idx][current_stops[end]], school)
    feasible =
        time_diff <= itinerary.slack_times[school_pos][end] &&
        travel_time(data, new_stop, school) <= max_travel_time(data, new_stop) &&
        (school_pos == 1 || data.schools[itinerary.schools[school_pos - 1]].start_time + itinerary.route_times[school_pos] + time_diff + school.dwell_time <= school.start_time)
    if feasible && time_diff < best_time_diff
        best_time_diff = time_diff
        insert_stop = length(current_stops)
    end
    return school_pos, insert_stop, best_time_diff
end


function itinerary_best_insertion_window(data::BirdData, school_idx::Int, new_stop_idx::Int, itinerary::Itinerary)
    school_pos = findfirst(==(school_idx), itinerary.schools)
    base_cost = itinerary_chain_time(data, itinerary)
    best_cost = Inf
    best_insert_school = -1
    best_insert_stop = -1

    if school_pos === nothing
        for insert_school in 0:length(itinerary.schools)
            candidate = insert_itinerary(data, itinerary, school_idx, new_stop_idx, insert_school, -1)
            itinerary_feasible(data, candidate) || continue
            cost = itinerary_chain_time(data, candidate) - base_cost
            if cost < best_cost
                best_cost = cost
                best_insert_school = insert_school
            end
        end
        return best_insert_school, -1, best_cost
    end

    for insert_stop in 0:length(itinerary.stops[school_pos])
        candidate = insert_itinerary(data, itinerary, school_idx, new_stop_idx, school_pos, insert_stop)
        itinerary_feasible(data, candidate) || continue
        cost = itinerary_chain_time(data, candidate) - base_cost
        if cost < best_cost
            best_cost = cost
            best_insert_school = school_pos
            best_insert_stop = insert_stop
        end
    end
    return best_insert_school, best_insert_stop, best_cost
end


function itinerary_best_insertion(data::BirdData, school_idx::Int, new_stop_idx::Int, itinerary::Itinerary)
    if uses_original_dwell_timing(data)
        return itinerary_best_insertion_original(data, school_idx, new_stop_idx, itinerary)
    end
    return itinerary_best_insertion_window(data, school_idx, new_stop_idx, itinerary)
end


function insert_itinerary(data::BirdData, itinerary::Itinerary, school_idx::Int, stop_idx::Int, insert_school::Int, insert_stop::Int)
    schools = copy(itinerary.schools)
    stops = deepcopy(itinerary.stops)
    n_students = copy(itinerary.n_students)
    n_wheelchair = copy(itinerary.n_wheelchair)
    slack_times = deepcopy(itinerary.slack_times)
    stop_times = deepcopy(itinerary.stop_times)
    route_times = copy(itinerary.route_times)
    if insert_stop < 0
        stop = data.stops[school_idx][stop_idx]
        time_on_bus = travel_time(data, stop, data.schools[school_idx])
        insert!(schools, insert_school + 1, school_idx)
        insert!(stops, insert_school + 1, [stop_idx])
        insert!(n_students, insert_school + 1, stop.n_students)
        insert!(n_wheelchair, insert_school + 1, stop.n_wheelchair)
        insert!(slack_times, insert_school + 1, [max_travel_time(data, stop) - time_on_bus])
        insert!(stop_times, insert_school + 1, [time_on_bus])
        insert!(route_times, insert_school + 1, time_on_bus + stop_time(data, stop))
        return Itinerary(schools, stops, n_students, n_wheelchair, slack_times, stop_times, route_times)
    end

    new_stop = data.stops[school_idx][stop_idx]
    if insert_stop == 0
        next_stop = data.stops[school_idx][stops[insert_school][1]]
        new_stop_time = travel_time(data, new_stop, next_stop) + stop_time(data, next_stop) + stop_times[insert_school][1]
        time_diff = 0.0
    elseif insert_stop == length(stops[insert_school])
        previous_stop = data.stops[school_idx][stops[insert_school][end]]
        new_stop_time = travel_time(data, new_stop, data.schools[school_idx])
        time_diff = new_stop_time + travel_time(data, previous_stop, new_stop) - travel_time(data, previous_stop, data.schools[school_idx])
    else
        previous_stop = data.stops[school_idx][stops[insert_school][insert_stop]]
        next_stop = data.stops[school_idx][stops[insert_school][insert_stop + 1]]
        time_to_next = travel_time(data, new_stop, next_stop)
        time_diff = travel_time(data, previous_stop, new_stop) + time_to_next - travel_time(data, previous_stop, next_stop)
        new_stop_time = time_to_next + stop_times[insert_school][insert_stop + 1] + stop_time(data, next_stop)
    end
    stop_times[insert_school] = vcat(
        stop_times[insert_school][1:insert_stop] .+ stop_time(data, new_stop) .+ time_diff,
        [new_stop_time],
        stop_times[insert_school][(insert_stop + 1):end],
    )
    slack_times[insert_school] = vcat(
        slack_times[insert_school][1:insert_stop] .- stop_time(data, new_stop) .- time_diff,
        [max_travel_time(data, new_stop) - new_stop_time],
        slack_times[insert_school][(insert_stop + 1):end],
    )
    for idx in eachindex(slack_times[insert_school])
        if idx > 1
            slack_times[insert_school][idx] = min(slack_times[insert_school][idx - 1], slack_times[insert_school][idx])
        end
    end
    stops[insert_school] = vcat(stops[insert_school][1:insert_stop], [stop_idx], stops[insert_school][(insert_stop + 1):end])
    n_students[insert_school] += new_stop.n_students
    n_wheelchair[insert_school] += new_stop.n_wheelchair
    route_times[insert_school] = stop_times[insert_school][1] + stop_time(data, data.stops[school_idx][stops[insert_school][1]])
    return Itinerary(schools, stops, n_students, n_wheelchair, slack_times, stop_times, route_times)
end


function solve_lbh!(data::BirdData; seed::Int = 1)
    data.fleet_aware && return solve_fleet_aware!(data; seed = seed)

    rng = MersenneTwister(seed)
    buses = BirdBus[]
    routes = [BirdRoute[] for _ in data.schools]
    available = [trues(length(data.stops[idx])) for idx in eachindex(data.schools)]
    use_original_timing = uses_original_dwell_timing(data)
    while any(any(mask) for mask in available)
        school_idx, stop_idx = random_stop(available; rng = rng)
        itinerary = initial_itinerary(data, school_idx, stop_idx)
        available[school_idx][stop_idx] = false
        while true
            best_school = 0
            best_stop = 0
            best_insert = (-1, -1)
            best_cost = Inf
            for candidate_school in eachindex(available)
                for candidate_stop in findall(identity, available[candidate_school])
                    stop_matches_itinerary_grade(data, itinerary, candidate_school, candidate_stop) || continue
                    under_capacity(data, itinerary, candidate_school, candidate_stop) || continue
                    insert_school, insert_stop, cost =
                        use_original_timing ?
                        itinerary_best_insertion_original(data, candidate_school, candidate_stop, itinerary) :
                        itinerary_best_insertion_window(data, candidate_school, candidate_stop, itinerary)
                    if cost < best_cost
                        best_cost = cost
                        best_school = candidate_school
                        best_stop = candidate_stop
                        best_insert = (insert_school, insert_stop)
                    end
                end
            end
            if isfinite(best_cost)
                itinerary = insert_itinerary(data, itinerary, best_school, best_stop, best_insert[1], best_insert[2])
                available[best_school][best_stop] = false
            else
                bus_schools = Int[]
                bus_routes = Int[]
                for (offset, school) in enumerate(itinerary.schools)
                    push!(routes[school], BirdRoute(length(routes[school]) + 1, itinerary.stops[offset]))
                    push!(bus_schools, school)
                    push!(bus_routes, length(routes[school]))
                end
                depot_id = data.depots[1].id
                arrival_times = itinerary_arrival_times(data, itinerary)
                arrival_times === nothing && error("internal Bird LBH timing failure")
                push!(buses, BirdBus(length(buses) + 1, depot_id, bus_schools, bus_routes, arrival_times))
                break
            end
        end
    end
    data.routes = routes
    data.scenarios = [[BirdScenario(idx, 1, collect(eachindex(routes[idx])))] for idx in eachindex(data.schools)]
    data.used_scenario = ones(Int, length(data.schools))
    data.buses = buses
    return data
end


function default_scenario_parameters(data::BirdData)
    upper = max(data.params.max_time_on_bus, 1.0)
    lower = max(upper / 2, 1.0)
    return [BirdScenarioParameters(lower, upper, 8, data.default_lambda_value, 8)]
end


function solve_with_scenarios!(
    data::BirdData;
    scenario_params = default_scenario_parameters(data),
    seed::Int = 1,
    optimizer = Gurobi.Optimizer,
    optimizer_attributes = Pair{String, Any}[],
)
    data.fleet_aware && return error("Scenarios not implemented for fleet aware")

    compute_scenarios!(data, scenario_params; seed = seed, optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    route_buses!(data; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    return data
end
