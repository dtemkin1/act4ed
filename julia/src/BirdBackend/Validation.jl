function route_passenger_load(data::BirdData, school_idx::Int, route::BirdRoute)
    return sum(data.stops[school_idx][stop_idx].n_students for stop_idx in route.stops)
end


function route_wheelchair_load(data::BirdData, school_idx::Int, route::BirdRoute)
    return sum(data.stops[school_idx][stop_idx].n_wheelchair for stop_idx in route.stops)
end


function test_feasibility(data::BirdData)
    unassigned_by_school = [Set{Int}() for _ in data.schools]
    for (school_idx, stop_idx) in data.unassigned_stops
        push!(unassigned_by_school[school_idx], stop_idx)
    end
    for school_idx in eachindex(data.routes)
        all_stops = Set(1:length(data.stops[school_idx]))
        for (route_idx, route) in enumerate(data.routes[school_idx])
            route_idx == route.id || error("route id mismatch for school $(school_idx)")
            isempty(route.stops) && error("route $(route_idx) for school $(school_idx) is empty")
            if !data.fleet_aware
                riders = route_passenger_load(data, school_idx, route)
                riders <= data.params.bus_capacity || error("route $(route_idx) for school $(school_idx) is over capacity")
            end
            time_on_bus = travel_time(data, data.stops[school_idx][route.stops[end]], data.schools[school_idx])
            time_on_bus <= max_travel_time(data, data.stops[school_idx][route.stops[end]]) || error("ride-time violation")
            time_on_bus += stop_time(data, data.stops[school_idx][route.stops[end]])
            next_stop = route.stops[end]
            if length(route.stops) > 1
                for reverse_pos in (length(route.stops) - 1):-1:1
                    stop_idx = route.stops[reverse_pos]
                    time_on_bus += travel_time(data, data.stops[school_idx][stop_idx], data.stops[school_idx][next_stop])
                    time_on_bus <= max_travel_time(data, data.stops[school_idx][stop_idx]) || error("ride-time violation")
                    time_on_bus += stop_time(data, data.stops[school_idx][stop_idx])
                    next_stop = stop_idx
                end
            end
        end
        if !isempty(data.scenarios)
            for scenario in data.scenarios[school_idx]
                covered = Set{Int}()
                for route_id in scenario.route_ids
                    for stop_idx in data.routes[school_idx][route_id].stops
                        stop_idx in covered && error("stop $(stop_idx) for school $(school_idx) visited twice")
                        push!(covered, stop_idx)
                    end
                end
                expected = data.allow_partial ? setdiff(all_stops, unassigned_by_school[school_idx]) : all_stops
                covered == expected || error("scenario $(scenario.id) does not cover expected stops for school $(school_idx)")
            end
        end
    end

    if !isempty(data.buses)
        routes_to_cover = [Set(data.scenarios[idx][data.used_scenario[idx]].route_ids) for idx in eachindex(data.schools)]
        for bus in data.buses
            isempty(bus.routes) && error("bus $(bus.id) serves no routes")
            length(bus.schools) == length(bus.routes) || error("bus $(bus.id) has inconsistent route data")
            isempty(bus.arrival_times) || length(bus.arrival_times) == length(bus.routes) || error("bus $(bus.id) has inconsistent arrival times")
            length(unique(bus.schools)) == length(bus.schools) || error("bus $(bus.id) repeats a school")
            for idx in eachindex(bus.schools)
                if !isempty(bus.arrival_times)
                    school = data.schools[bus.schools[idx]]
                    arrival_time = bus.arrival_times[idx]
                    earliest_arrival_time(school) - BIRD_TIMING_EPS <= arrival_time || error("bus $(bus.id) arrives too early at school $(bus.schools[idx])")
                    arrival_time <= latest_arrival_time(school) + BIRD_TIMING_EPS || error("bus $(bus.id) arrives too late at school $(bus.schools[idx])")
                    if idx > 1
                        previous_school = data.schools[bus.schools[idx - 1]]
                        route = data.routes[bus.schools[idx]][bus.routes[idx]]
                        first_stop = data.stops[bus.schools[idx]][route.stops[1]]
                        earliest_feasible_arrival =
                            bus.arrival_times[idx - 1] +
                            travel_time(data, previous_school, first_stop) +
                            service_time(data, bus.schools[idx], route)
                        arrival_time + BIRD_TIMING_EPS >= earliest_feasible_arrival || error("bus $(bus.id) has an infeasible school-to-school transfer")
                    end
                end
                if data.fleet_aware
                    bus.id in eachindex(data.fleet) || error("bus $(bus.id) is not in the concrete fleet")
                    fleet_bus = data.fleet[bus.id]
                    bus.depot == fleet_bus.depot || error("bus $(bus.id) uses depot $(bus.depot), expected $(fleet_bus.depot)")
                    route = data.routes[bus.schools[idx]][bus.routes[idx]]
                    riders = route_passenger_load(data, bus.schools[idx], route)
                    wheelchair_riders = route_wheelchair_load(data, bus.schools[idx], route)
                    riders <= fleet_bus.capacity || error("route $(route.id) for school $(bus.schools[idx]) exceeds capacity for bus $(fleet_bus.name)")
                    wheelchair_riders <= fleet_bus.wheelchair_capacity || error("route $(route.id) for school $(bus.schools[idx]) exceeds wheelchair capacity for bus $(fleet_bus.name)")
                end
                delete!(routes_to_cover[bus.schools[idx]], bus.routes[idx])
            end
        end
        for school_idx in eachindex(data.schools)
            isempty(routes_to_cover[school_idx]) || error("school $(school_idx) has uncovered routes")
        end
    end
    return true
end
