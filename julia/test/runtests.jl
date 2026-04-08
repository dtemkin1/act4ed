using Test
using JuMP
using SparseArrays

include("../src/Formulation3JuMP.jl")
using .Formulation3JuMP


@testset "Formulation3JuMP" begin
    path = normpath(joinpath(@__DIR__, "..", "testdata", "tiny_instance.npz"))
    inst = load_instance(path)

    @test inst.nB == 1
    @test inst.nM == 2
    @test inst.nQ == 2
    @test nnz(inst.student_to_pickup) == 2
    @test inst.pickup_node_of_m == [1, 1]

    bundle = build_model(inst; optimizer = nothing)
    @test JuMP.num_variables(bundle.model) == 70
    @test :x_bqij in keys(bundle.vars)
    @test length(bundle.meta.bus_names) == 1
    @test length(bundle.meta.node_ids) == inst.nN
end
