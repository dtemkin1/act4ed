function test_feasibility(data::BirdData)
    for school_idx in eachindex(data.routes)
        all_stops = Set(1:length(data.stops[school_idx]))
        for (route_idx, route) in enumerate(data.routes[school_idx])
            route_idx == route.id || error("route id mismatch for school $(school_idx)")
            isempty(route.stops) && error("route $(route_idx) for school $(school_idx) is empty")
            riders = sum(data.stops[school_idx][stop_idx].n_students for stop_idx in route.stops)
            riders <= data.params.bus_capacity || error("route $(route_idx) for school $(school_idx) is over capacity")
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
                covered == all_stops || error("scenario $(scenario.id) does not cover all stops for school $(school_idx)")
            end
        end
    end

    if !isempty(data.buses)
        routes_to_cover = [Set(data.scenarios[idx][data.used_scenario[idx]].route_ids) for idx in eachindex(data.schools)]
        for bus in data.buses
            isempty(bus.routes) && error("bus $(bus.id) serves no routes")
            length(bus.schools) == length(bus.routes) || error("bus $(bus.id) has inconsistent route data")
            length(unique(bus.schools)) == length(bus.schools) || error("bus $(bus.id) repeats a school")
            for idx in eachindex(bus.schools)
                delete!(routes_to_cover[bus.schools[idx]], bus.routes[idx])
            end
        end
        for school_idx in eachindex(data.schools)
            isempty(routes_to_cover[school_idx]) || error("school $(school_idx) has uncovered routes")
        end
    end
    return true
end
