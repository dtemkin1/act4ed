include("../julia/src/BirdBackend.jl")
using .BirdBackend


function parse_args(args)
    parsed = Dict{String, String}()
    index = 1
    while index <= length(args)
        key = args[index]
        if !(key in ("--instance", "--solution", "--log-file", "--method", "--seed"))
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
            "usage: solve_bird_backend_julia.jl --instance <path> --solution <path> [--log-file <path>] [--method lbh|scenario] [--seed <int>]",
        )
    end
    return parsed
end


function main(args = ARGS)
    parsed = parse_args(args)
    instance_path = parsed["--instance"]
    solution_path = parsed["--solution"]
    method = get(parsed, "--method", "scenario")
    log_file = get(parsed, "--log-file", nothing)
    seed = parse(Int, get(parsed, "--seed", "1"))

    data = load_instance(instance_path)
    optimizer_attributes =
        log_file === nothing ? Pair{String, Any}[] : Pair{String, Any}["LogFile" => log_file]

    start_time = time()
    if method == "lbh"
        solve_lbh!(data; seed = seed)
    elseif method == "scenario"
        solve_with_scenarios!(
            data;
            seed = seed,
            optimizer_attributes = optimizer_attributes,
        )
    else
        error("unknown Bird solve method: $(method)")
    end
    runtime_seconds = time() - start_time

    test_feasibility(data)
    solution = snapshot_solution(
        data;
        runtime_seconds = runtime_seconds,
        status_name = "OPTIMAL",
        objective_value = float(length(data.buses)),
    )
    save_solution(solution_path, solution)

    summary = Dict(
        "instance_path" => instance_path,
        "solution_path" => solution_path,
        "method" => method,
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
