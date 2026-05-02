module BirdBackend

using JuMP
using Gurobi
using MathOptInterface
using NPZ
using Random

export BirdParameters,
    BirdDemandStop,
    BirdSchool,
    BirdDepot,
    BirdRoute,
    BirdScenario,
    BirdBus,
    BirdFleetBus,
    BirdData,
    BirdScenarioParameters,
    BirdBackendSolution,
    load_instance,
    load_legacy_benchmark,
    compute_scenarios!,
    route_buses!,
    solve_fleet_aware!,
    solve_fleet_aware_with_scenarios!,
    solve_with_scenarios!,
    solve_lbh!,
    snapshot_solution,
    save_solution,
    test_feasibility

include("BirdBackend/Core.jl")
include("BirdBackend/Loading.jl")
include("BirdBackend/Scenarios.jl")
include("BirdBackend/Routing.jl")
include("BirdBackend/FleetScenarios.jl")
include("BirdBackend/LBH.jl")
include("BirdBackend/Validation.jl")
include("BirdBackend/SolutionIO.jl")

end
