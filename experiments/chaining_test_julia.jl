using JuMP

include("../julia/src/Formulation3JuMP.jl")
using .Formulation3JuMP

const CURRENT_FILE_DIR = @__DIR__
const OUTPUT_DIR = joinpath(CURRENT_FILE_DIR, "outputs")

function export_instance!(instance_path::AbstractString; rounds::Int)
    mkpath(dirname(instance_path))
    cmd = `uv run python -m experiments.chaining_test_julia_export --rounds $rounds --output $instance_path`
    run(cmd)
    return instance_path
end

function summary_text(; problem_name, rounds, status, objective_value, runtime_seconds, num_variables, num_constraints)
    return join(
        [
            "problem_name: $(problem_name)",
            "rounds: $(rounds)",
            "status: $(status)",
            "objective_value: $(objective_value)",
            "runtime_seconds: $(runtime_seconds)",
            "num_variables: $(num_variables)",
            "num_constraints: $(num_constraints)",
        ],
        "\n",
    )
end

function main(args = ARGS)
    rounds = isempty(args) ? 1 : parse(Int, args[1])
    problem_name = "chaining_test"

    instance_path = joinpath(OUTPUT_DIR, "$(problem_name)_rounds_$(rounds).npz")
    export_instance!(instance_path; rounds = rounds)

    inst = load_instance(instance_path)
    bundle = build_model(inst)

    println("Model built for problem: $(problem_name)")
    optimize!(bundle.model)
    println("Problem solved: $(problem_name)")

    solution_path = joinpath(OUTPUT_DIR, "$(problem_name)_rounds_$(rounds)_julia_solution.npz")
    save_solution(solution_path, bundle)
    println("Wrote solution to $(solution_path)")

    status = termination_status(bundle.model)
    obj_value = has_values(bundle.model) ? JuMP.objective_value(bundle.model) : missing
    runtime_seconds = solve_time(bundle.model)
    num_variables = JuMP.num_variables(bundle.model)
    num_constraints = JuMP.num_constraints(bundle.model; count_variable_in_set_constraints = false)

    summary = Dict(
        "problem_name" => problem_name,
        "rounds" => rounds,
        "status" => string(status),
        "objective_value" => obj_value,
        "runtime_seconds" => runtime_seconds,
        "num_variables" => num_variables,
        "num_constraints" => num_constraints,
        "instance_path" => instance_path,
    )
    println(summary)

    summary_path = joinpath(OUTPUT_DIR, "$(problem_name)_rounds_$(rounds)_julia.txt")
    open(summary_path, "w") do io
        write(
            io,
            summary_text(
                problem_name = problem_name,
                rounds = rounds,
                status = status,
                objective_value = obj_value,
                runtime_seconds = runtime_seconds,
                num_variables = num_variables,
                num_constraints = num_constraints,
            ),
        )
    end
    println("Wrote summary to $(summary_path)")

    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
