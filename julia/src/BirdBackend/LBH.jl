struct Itinerary
    schools::Vector{Int}
    stops::Vector{Vector{Int}}
    n_students::Vector{Int}
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


function under_capacity(data::BirdData, itinerary::Itinerary, school_idx::Int, stop_idx::Int)
    current_idx = findfirst(==(school_idx), itinerary.schools)
    current_idx === nothing && return true
    return data.stops[school_idx][stop_idx].n_students + itinerary.n_students[current_idx] <= data.params.bus_capacity
end


function initial_itinerary(data::BirdData, school_idx::Int, stop_idx::Int)
    stop = data.stops[school_idx][stop_idx]
    school = data.schools[school_idx]
    time_on_bus = travel_time(data, stop, school)
    return Itinerary(
        [school_idx],
        [[stop_idx]],
        [stop.n_students],
        [[max_travel_time(data, stop) - time_on_bus]],
        [[time_on_bus]],
        [time_on_bus + stop_time(data, stop)],
    )
end


function itinerary_best_insertion(data::BirdData, school_idx::Int, new_stop_idx::Int, itinerary::Itinerary)
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


function insert_itinerary(data::BirdData, itinerary::Itinerary, school_idx::Int, stop_idx::Int, insert_school::Int, insert_stop::Int)
    schools = copy(itinerary.schools)
    stops = deepcopy(itinerary.stops)
    n_students = copy(itinerary.n_students)
    slack_times = deepcopy(itinerary.slack_times)
    stop_times = deepcopy(itinerary.stop_times)
    route_times = copy(itinerary.route_times)
    if insert_stop < 0
        stop = data.stops[school_idx][stop_idx]
        time_on_bus = travel_time(data, stop, data.schools[school_idx])
        insert!(schools, insert_school + 1, school_idx)
        insert!(stops, insert_school + 1, [stop_idx])
        insert!(n_students, insert_school + 1, stop.n_students)
        insert!(slack_times, insert_school + 1, [max_travel_time(data, stop) - time_on_bus])
        insert!(stop_times, insert_school + 1, [time_on_bus])
        insert!(route_times, insert_school + 1, time_on_bus + stop_time(data, stop))
        return Itinerary(schools, stops, n_students, slack_times, stop_times, route_times)
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
    route_times[insert_school] = stop_times[insert_school][1] + stop_time(data, data.stops[school_idx][stops[insert_school][1]])
    return Itinerary(schools, stops, n_students, slack_times, stop_times, route_times)
end


function solve_lbh!(data::BirdData; seed::Int = 1)
    rng = MersenneTwister(seed)
    buses = BirdBus[]
    routes = [BirdRoute[] for _ in data.schools]
    available = [trues(length(data.stops[idx])) for idx in eachindex(data.schools)]
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
                    under_capacity(data, itinerary, candidate_school, candidate_stop) || continue
                    insert_school, insert_stop, cost = itinerary_best_insertion(data, candidate_school, candidate_stop, itinerary)
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
                push!(buses, BirdBus(length(buses) + 1, depot_id, bus_schools, bus_routes))
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
    compute_scenarios!(data, scenario_params; seed = seed, optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    route_buses!(data; optimizer = optimizer, optimizer_attributes = optimizer_attributes)
    return data
end
