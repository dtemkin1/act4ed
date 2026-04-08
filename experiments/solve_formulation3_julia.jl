using JuMP

include("../julia/src/Formulation3JuMP.jl")
using .Formulation3JuMP


function parse_args(args)
    parsed = Dict{String, String}()
    index = 1
    while index <= length(args)
        key = args[index]
        if !(key in ("--instance", "--solution", "--log-file"))
            error("unknown argument: $(key)")
        end
        if index == length(args)
            error("missing value for $(key)")
        end
        parsed[key] = args[index + 1]
        index += 2
    end
    if !haskey(parsed, "--instance") || !haskey(parsed, "--solution")
        error("usage: solve_formulation3_julia.jl --instance <path> --solution <path> [--log-file <path>]")
    end
    return parsed
end


function main(args = ARGS)
    parsed = parse_args(args)
    instance_path = parsed["--instance"]
    solution_path = parsed["--solution"]
    log_file = get(parsed, "--log-file", nothing)

    inst = load_instance(instance_path)
    bundle = build_model(inst; log_file = log_file)

    println("Model built from $(instance_path)")
    optimize!(bundle.model)
    println("Problem solved")

    save_solution(solution_path, bundle)
    println("Wrote solution to $(solution_path)")

    status = termination_status(bundle.model)
    objective = has_values(bundle.model) ? objective_value(bundle.model) : missing
    summary = Dict(
        "instance_path" => instance_path,
        "solution_path" => solution_path,
        "status" => string(status),
        "objective_value" => objective,
        "runtime_seconds" => solve_time(bundle.model),
        "num_variables" => JuMP.num_variables(bundle.model),
        "num_constraints" => JuMP.num_constraints(
            bundle.model;
            count_variable_in_set_constraints = false,
        ),
    )
    println(summary)
    return summary
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
