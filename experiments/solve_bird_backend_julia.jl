include("../julia/src/BirdBackend.jl")
using .BirdBackend


function parse_args(args)
    parsed = Dict{String, String}()
    index = 1
    flag_args = Set(["--gurobi-verbose", "--timing-log"])
    value_args = Set(["--instance", "--solution", "--log-file", "--seed", "--gurobi-threads"])
    while index <= length(args)
        key = args[index]
        if key in flag_args
            parsed[key] = "true"
            index += 1
            continue
        end
        if !(key in value_args)
            error("unknown argument: $(key)")
        end
        if index == length(args)
            error("missing value for $(key)")
        end
        parsed[key] = args[index + 1]
        index += 2
    end
    if !haskey(parsed, "--instance") || !haskey(parsed, "--solution")
        error(
            "usage: solve_bird_backend_julia.jl --instance <path> --solution <path> [--log-file <path>] [--seed <int>] [--gurobi-threads <int>] [--timing-log] [--gurobi-verbose]",
        )
    end
    return parsed
end


function main(args = ARGS)
    parsed = parse_args(args)
    instance_path = parsed["--instance"]
    solution_path = parsed["--solution"]
    log_file = get(parsed, "--log-file", nothing)
    seed = parse(Int, get(parsed, "--seed", "1"))
    gurobi_threads = parse(Int, get(parsed, "--gurobi-threads", "0"))
    gurobi_verbose = get(parsed, "--gurobi-verbose", "false") == "true"
    timing_log = get(parsed, "--timing-log", "false") == "true"
    gurobi_threads < 0 && error("--gurobi-threads must be nonnegative")

    data = load_instance(instance_path)
    optimizer_attributes = Pair{String, Any}[]
    if log_file !== nothing
        push!(optimizer_attributes, "LogFile" => log_file)
    end
    if gurobi_verbose
        push!(optimizer_attributes, "OutputFlag" => 1)
    end
    if gurobi_threads > 0
        push!(optimizer_attributes, "Threads" => gurobi_threads)
    end

    start_time = time()
    if data.method == "lbh"
        solve_lbh!(data; seed = seed)
    elseif data.method == "scenario"
        solve_with_scenarios!(
            data;
            seed = seed,
            optimizer_attributes = optimizer_attributes,
            timing_log = timing_log,
        )
    else
        error("unknown Bird solve method: $(data.method)")
    end
    runtime_seconds = time() - start_time

    test_feasibility(data)
    status_name = data.allow_partial && !isempty(data.unassigned_stops) ? "PARTIAL" : "OPTIMAL"
    solution = snapshot_solution(
        data;
        runtime_seconds = runtime_seconds,
        status_name = status_name,
        objective_value = float(length(data.buses)),
    )
    save_solution(solution_path, solution)

    summary = Dict(
        "instance_path" => instance_path,
        "solution_path" => solution_path,
        "method" => data.method,
        "status" => solution.status_name,
        "runtime_seconds" => runtime_seconds,
        "buses_used" => solution.buses_used,
        "total_distance_km" => solution.total_distance_km,
    )
    println(summary)
    return summary
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
