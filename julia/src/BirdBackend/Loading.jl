function load_legacy_benchmark(
    schools_path::AbstractString,
    stops_path::AbstractString;
    bus_capacity::Int = 66,
    max_time_on_bus::Float64 = 45.0,
    constant_stop_time::Float64 = 19.0 / 60.0,
    stop_time_per_student::Float64 = 2.6 / 60.0,
    stop_time_per_wheelchair_student::Float64 = 0.0,
    speed_units_per_minute::Float64 = 29.3333333 * 60.0,
    school_dwell_time::Float64 = 154.4 / 60.0,
)
    school_rows = _read_tsv(schools_path)
    stop_rows = _read_tsv(stops_path)

    coords = Vector{Tuple{Float64, Float64}}()
    schools = BirdSchool[]
    school_id_map = Dict{String, Int}()
    for row in school_rows
        push!(coords, (parse(Float64, row["X"]), parse(Float64, row["Y"])))
        school_idx = length(schools) + 1
        push!(
            schools,
            BirdSchool(
                school_idx,
                row["ID"],
                row["ID"],
                _school_start_time(row["AMEARLY"]),
                school_dwell_time,
                school_dwell_time,
                school_dwell_time,
                0,
            ),
        )
        school_id_map[row["ID"]] = school_idx
    end

    depots = [BirdDepot(1, "legacy-depot", "legacy-depot", 0)]
    grouped_stops = [BirdDemandStop[] for _ in schools]
    stop_coords = Tuple{Float64, Float64}[]
    for row in stop_rows
        school_idx = school_id_map[row["EP_ID"]]
        push!(stop_coords, (parse(Float64, row["X_COORD"]), parse(Float64, row["Y_COORD"])))
        push!(
            grouped_stops[school_idx],
            BirdDemandStop(
                length(grouped_stops[school_idx]) + 1,
                sum(length.(grouped_stops)) + 1,
                row["ID"],
                row["ID"],
                school_idx,
                0,
                parse(Int, row["STUDENT_COUNT"]),
                0,
                SERVICE_GROUP_CONVENTIONAL,
                1,
            ),
        )
    end

    all_coords = Tuple{Float64, Float64}[]
    for school_idx in eachindex(grouped_stops)
        for stop in grouped_stops[school_idx]
            row = stop_coords[stop.unique_id]
            push!(all_coords, row)
        end
    end
    append!(all_coords, coords)
    push!(all_coords, (105_600.0, 105_600.0))

    stop_count = sum(length.(grouped_stops))
    for school_idx in eachindex(schools)
        schools[school_idx] = BirdSchool(
            schools[school_idx].id,
            schools[school_idx].external_id,
            schools[school_idx].name,
            schools[school_idx].start_time,
            schools[school_idx].dwell_time,
            schools[school_idx].earliest_arrival_buffer,
            schools[school_idx].latest_arrival_buffer,
            stop_count + school_idx,
        )
    end
    depots[1] = BirdDepot(1, depots[1].external_id, depots[1].name, stop_count + length(schools) + 1)

    global_stop_idx = 1
    for school_idx in eachindex(grouped_stops)
        for local_idx in eachindex(grouped_stops[school_idx])
            stop = grouped_stops[school_idx][local_idx]
            grouped_stops[school_idx][local_idx] = BirdDemandStop(
                stop.id,
                global_stop_idx,
                stop.external_id,
                stop.source_stop_id,
                stop.school_id,
                global_stop_idx,
                stop.n_students,
                stop.n_wheelchair,
                stop.group_id,
                stop.grade_id,
            )
            global_stop_idx += 1
        end
    end

    node_count = length(all_coords)
    travel_time_min = fill(Inf, node_count, node_count)
    travel_distance = fill(Inf, node_count, node_count)
    for src in 1:node_count
        travel_time_min[src, src] = 0.0
        travel_distance[src, src] = 0.0
        for dst in 1:node_count
            src == dst && continue
            distance = abs(all_coords[src][1] - all_coords[dst][1]) + abs(all_coords[src][2] - all_coords[dst][2])
            travel_distance[src, dst] = distance
            travel_time_min[src, dst] = distance / speed_units_per_minute
        end
    end

    return BirdData(
        BirdParameters(
            bus_capacity,
            max_time_on_bus,
            constant_stop_time,
            stop_time_per_student,
            stop_time_per_wheelchair_student,
        ),
        schools,
        depots,
        grouped_stops,
        travel_time_min,
        travel_distance,
        bus_capacity,
        typemax(Int),
        "legacy",
        "legacy",
        false,
        false,
        false,
        BirdFleetBus[],
        DEFAULT_LAMBDA_VALUE,
        [BirdScenario[] for _ in schools],
        [BirdRoute[] for _ in schools],
        zeros(Int, length(schools)),
        BirdBus[],
        Tuple{Int, Int}[],
    )
end


function load_instance(path::AbstractString)
    schema_data = NPZ.npzread(path, ["schema_version"])
    schema_version = _scalar(schema_data, "schema_version", Int)
    schema_version in 1:BIRD_INSTANCE_SCHEMA_VERSION || error("unsupported Bird instance schema")

    keys = [
        "schema_version",
        "max_time_on_bus",
        "constant_stop_time",
        "stop_time_per_student",
        "bus_capacity",
        "fleet_size",
        "school_start_times",
        "school_dwell_times",
        "demand_school_indices",
        "demand_students",
        "travel_distance_km",
        "travel_time_min",
    ]
    if schema_version >= 2
        push!(keys, "lambda_value")
    end
    if schema_version >= 5
        append!(
            keys,
            [
                "fleet_aware",
                "conventional_spillover",
                "bus_capacities",
                "bus_depot_indices",
                "bus_has_monitor",
                "bus_wheelchair_capacities",
                "demand_group_ids",
                "demand_wheelchair_students",
            ],
        )
    end
    if schema_version >= 6
        push!(keys, "allow_partial")
    end
    if schema_version >= 7
        append!(keys, ["school_earliest_arrival_buffers", "school_latest_arrival_buffers"])
    end
    if schema_version >= 8
        push!(keys, "stop_time_per_wheelchair_student")
    end
    if schema_version >= 9
        push!(keys, "demand_grade_ids")
    end
    data = NPZ.npzread(path, keys)
    default_lambda_value = schema_version >= 2 ? _scalar(data, "lambda_value", Float64) : DEFAULT_LAMBDA_VALUE

    school_start_times = _fvec(data, "school_start_times")
    school_dwell_times = _fvec(data, "school_dwell_times")
    school_latest_arrival_buffers =
        schema_version >= 7 ? _fvec(data, "school_latest_arrival_buffers") : copy(school_dwell_times)
    school_earliest_arrival_buffers =
        schema_version >= 7 ? _fvec(data, "school_earliest_arrival_buffers") : copy(school_latest_arrival_buffers)
    demand_school_indices = _ivec(data, "demand_school_indices")
    demand_students = _ivec(data, "demand_students")
    demand_group_ids =
        schema_version >= 5 ? _ivec(data, "demand_group_ids") : fill(SERVICE_GROUP_CONVENTIONAL, length(demand_students))
    demand_wheelchair_students =
        schema_version >= 5 ? _ivec(data, "demand_wheelchair_students") : zeros(Int, length(demand_students))
    demand_grade_ids =
        schema_version >= 9 ? _ivec(data, "demand_grade_ids") : ones(Int, length(demand_students))
    cohort = "exported"
    bus_type = "exported"

    demand_count = length(demand_school_indices)
    schools = [
        BirdSchool(
            idx,
            "school_$(idx)",
            "school_$(idx)",
            school_start_times[idx],
            school_dwell_times[idx],
            school_earliest_arrival_buffers[idx],
            school_latest_arrival_buffers[idx],
            demand_count + idx,
        ) for idx in eachindex(school_start_times)
    ]
    depot_count = size(data["travel_time_min"], 1) - demand_count - length(schools)
    depots = [BirdDepot(idx, "depot_$(idx)", "depot_$(idx)", demand_count + length(schools) + idx) for idx in 1:depot_count]

    stops = [BirdDemandStop[] for _ in schools]
    for row_idx in eachindex(demand_school_indices)
        school_idx = demand_school_indices[row_idx]
        push!(
            stops[school_idx],
            BirdDemandStop(
                length(stops[school_idx]) + 1,
                row_idx,
                "stop_$(row_idx)",
                "stop_$(row_idx)",
                school_idx,
                row_idx,
                demand_students[row_idx],
                demand_wheelchair_students[row_idx],
                demand_group_ids[row_idx],
                demand_grade_ids[row_idx],
            ),
        )
    end
    fleet_aware =
        schema_version >= 5 ? _scalar(data, "fleet_aware", Int) == 1 : false
    conventional_spillover =
        schema_version >= 5 ? _scalar(data, "conventional_spillover", Int) == 1 : false
    allow_partial =
        schema_version >= 6 ? _scalar(data, "allow_partial", Int) == 1 : false
    fleet = BirdFleetBus[]
    if schema_version >= 5
        bus_capacities = _ivec(data, "bus_capacities")
        bus_depot_indices = _ivec(data, "bus_depot_indices")
        bus_has_monitor = _ivec(data, "bus_has_monitor")
        bus_wheelchair_capacities = _ivec(data, "bus_wheelchair_capacities")
        for idx in eachindex(bus_capacities)
            push!(
                fleet,
                BirdFleetBus(
                    idx,
                    "bus_$(idx)",
                    bus_depot_indices[idx],
                    bus_capacities[idx],
                    bus_has_monitor[idx] == 1,
                    bus_wheelchair_capacities[idx],
                    "exported",
                ),
            )
        end
    end

    return BirdData(
        BirdParameters(
            _scalar(data, "bus_capacity", Int),
            _scalar(data, "max_time_on_bus", Float64),
            _scalar(data, "constant_stop_time", Float64),
            _scalar(data, "stop_time_per_student", Float64),
            schema_version >= 8 ? _scalar(data, "stop_time_per_wheelchair_student", Float64) : 0.0,
        ),
        schools,
        depots,
        stops,
        Float64.(data["travel_time_min"]),
        Float64.(data["travel_distance_km"]),
        _scalar(data, "bus_capacity", Int),
        _scalar(data, "fleet_size", Int),
        cohort,
        bus_type,
        fleet_aware,
        conventional_spillover,
        allow_partial,
        fleet,
        default_lambda_value,
        [BirdScenario[] for _ in schools],
        [BirdRoute[] for _ in schools],
        zeros(Int, length(schools)),
        BirdBus[],
        Tuple{Int, Int}[],
    )
end
