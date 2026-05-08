using Test
using JuMP
using SparseArrays

include("../src/Formulation3JuMP.jl")
using .Formulation3JuMP
include("../src/BirdBackend.jl")
using .BirdBackend


@testset "Formulation3JuMP" begin
    path = normpath(joinpath(@__DIR__, "..", "testdata", "tiny_instance.npz"))
    inst = Formulation3JuMP.load_instance(path)

    @test inst.nB == 1
    @test inst.nM == 2
    @test inst.nQ == 2
    @test nnz(inst.student_to_pickup) == 2
    @test inst.pickup_node_of_m == [1, 1]

    bundle = Formulation3JuMP.build_model(inst; optimizer = nothing)
    @test JuMP.num_variables(bundle.model) == 70
    @test :x_bqij in keys(bundle.vars)
    @test length(bundle.meta.bus_names) == 1
    @test length(bundle.meta.node_ids) == inst.nN
end


function _stop_time_test_data()
    return BirdData(
        BirdParameters(66, 120.0, 1.0, 2.0, 3.0, 5.0),
        BirdSchool[],
        BirdDepot[],
        Vector{Vector{BirdDemandStop}}(),
        zeros(Float64, 0, 0),
        zeros(Float64, 0, 0),
        66,
        0,
        "test",
        "test",
        false,
        false,
        false,
        BirdFleetBus[],
        1.0e4,
        Vector{Vector{BirdScenario}}(),
        Vector{Vector{BirdRoute}}(),
        Int[],
        BirdBus[],
        Tuple{Int, Int}[],
        "lbh",
    )
end


function _stop_time_test_stop(n_students::Int; n_wheelchair::Int = 0, n_sped::Int = 0)
    return BirdDemandStop(
        1,
        1,
        "stop",
        "stop",
        1,
        1,
        n_students,
        n_wheelchair,
        n_sped,
        BirdBackend.SERVICE_GROUP_CONVENTIONAL,
        1,
    )
end


@testset "BirdBackend stop time" begin
    data = _stop_time_test_data()

    @test BirdBackend.stop_time(data, _stop_time_test_stop(2)) == 5.0
    @test BirdBackend.stop_time(data, _stop_time_test_stop(2; n_sped = 2)) == 15.0
    @test BirdBackend.stop_time(data, _stop_time_test_stop(2; n_wheelchair = 2)) == 11.0
    @test BirdBackend.stop_time(data, _stop_time_test_stop(1; n_wheelchair = 1, n_sped = 1)) == 8.0
end


@testset "BirdBackend" begin
    schools_path = normpath(joinpath(@__DIR__, "..", "..", "bird", "data", "input", "CSCB01", "Schools.txt"))
    stops_path = normpath(joinpath(@__DIR__, "..", "..", "bird", "data", "input", "CSCB01", "Stops.txt"))

    data = BirdBackend.load_legacy_benchmark(schools_path, stops_path)
    @test length(data.schools) > 0
    @test sum(length.(data.stops)) > 0
    @test size(data.travel_time_min, 1) == size(data.travel_time_min, 2)

    BirdBackend.solve_lbh!(data; seed = 1)
    @test BirdBackend.test_feasibility(data)

    solution = BirdBackend.snapshot_solution(data; runtime_seconds = 0.0)
    @test solution.status_name == "OPTIMAL"
    @test solution.buses_used == length(data.buses)
    @test solution.buses_used > 0
    @test length(solution.assignment_bus_ids) == length(solution.assignment_orders)
    @test length(solution.assignment_stop_ptr) == length(solution.assignment_bus_ids) + 1
end
