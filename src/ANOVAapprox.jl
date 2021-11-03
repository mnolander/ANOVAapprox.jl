module ANOVAapprox

using GroupedTransforms,
    LinearAlgebra, IterativeSolvers, LinearMaps, Distributed, SpecialFunctions

bases = ["per", "cos", "cheb", "std"]
types = Dict("per" => ComplexF64, "cos" => Float64, "cheb" => Float64, "std" => Float64)
vtypes = Dict(
    "per" => Vector{ComplexF64},
    "cos" => Vector{Float64},
    "cheb" => Vector{Float64},
    "std" => Vector{Float64},
)
gt_systems = Dict("per" => "exp", "cos" => "cos", "cheb" => "cos", "std" => "cos")

function get_orderDependentBW(U::Vector{Vector{Int}}, N::Vector{Int})::Vector{Int}
    N_bw = zeros(Int64, length(U))

    for i = 1:length(U)
        if U[i] == []
            N_bw[i] = 0
        else
            N_bw[i] = N[length(U[i])]
        end
    end

    return N_bw
end

include("fista.jl")
include("approx.jl")
include("errors.jl")
include("analysis.jl")

end # module
