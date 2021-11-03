module TestFunctionCheb

using LinearAlgebra
using Random

function B2(x::Float64)::Float64
    norm_b2_cheb = sqrt((-423 * sqrt(3) + 920 * pi) / (4096 * pi))

    if (x < -1) || (x > 1)
        error("Out of bounds.")
    end

    if x < -9 / 2
        y = 0.0
    elseif (x >= -9 / 2) && (x < -5 / 2)
        y = 1 / 32 * (2 * x + 9)^2
    elseif (x >= -5 / 2) && (x < -1 / 2)
        y = (3 / 16 - 3 / 4 * x - 1 / 4 * x^2)
    elseif (x >= -1 / 2) && (x < 3 / 2)
        y = (9 / 32 - 3 / 8 * x + 1 / 8 * x^2)
    elseif (x >= 3 / 2)
        y = 0.0
    end

    return y / norm_b2_cheb
end

function B4(x::Float64)::Float64
    norm_b4_cheb = sqrt((-36614943 * sqrt(3) + 218675240 * pi) / (6341787648 * pi))

    if (x < -1) || (x > 1)
        error("Out of bounds.")
    end

    if x < -15 / 2
        y = 0.0
    elseif (x >= -15 / 2) && (x < -11 / 2)
        y = 1 / 6144 * (2 * x + 15)^4
    elseif (x >= -11 / 2) && (x < -7 / 2)
        y = -5645 / 1536 - 205 / 48 * x - 95 / 64 * x^2 - 5 / 24 * x^3 - 1 / 96 * x^4
    elseif (x >= -7 / 2) && (x < -3 / 2)
        y = 715 / 3072 + 25 / 128 * x + 55 / 128 * x^2 + 5 / 32 * x^3 + 1 / 64 * x^4
    elseif (x >= -3 / 2) && (x < 1 / 2)
        y = 155 / 1536 - 5 / 32 * x + 5 / 64 * x^2 - 1 / 96 * x^4
    elseif (x >= 1 / 2) && (x < 5 / 2)
        y = 1 / 6144 * (2 * x - 5)^4
    elseif (x >= 3 / 2)
        y = 0.0
    end

    return y / norm_b4_cheb
end

function B2_chat(k::Integer)::Float64
    norm_b2_cheb = sqrt((-423 * sqrt(3) + 920 * pi) / (4096 * pi))

    if k < 0
        error("Out of bounds.")
    end

    if k == 0
        fc = 1 / 4 + (9 * sqrt(3)) / (64 * pi)
    elseif k == 1
        fc = -1 / 2 + (9 * sqrt(3)) / (32 * pi)
    elseif k == 2
        fc = (9 * sqrt(3)) / (128 * pi)
    elseif k > 2
        fc =
            (
                9 * sqrt(3) * k * cos((2 * k * pi) / 3) -
                9 * (-2 + k^2) * sin((2 * k * pi) / 3)
            ) / (8 * k * (4 - 5 * k^2 + k^4) * pi)
    end

    if k != 0
        fc /= sqrt(2)
    end

    return fc / norm_b2_cheb
end

function B4_chat(k::Integer)::Float64
    norm_b4_cheb = sqrt((-36614943 * sqrt(3) + 218675240 * pi) / (6341787648 * pi))

    if k < 0
        error("Out of bounds.")
    end

    if k == 0
        fc = 2603 / 18432 - 75 / 8192 * sqrt(3) / pi
    elseif k == 1
        fc = -95 / 576 + 33 * sqrt(3) / (2048 * pi)
    elseif k == 2
        fc = 181 / 4608 - 39 * sqrt(3) / (4096 * pi)
    elseif k == 3
        fc = (5 * (-14 + (27 * sqrt(3)) / pi)) / 32256
    elseif k == 4
        fc = -7 / 9216 - 93 * sqrt(3) / (114688 * pi)
    elseif k > 4
        fc =
            (
                (900 * sqrt(3) * k * (-9 + k^2) * cos(k * pi) / 3) +
                90 * (152 - 75 * k^2 + 3 * k^4) * sin(k * pi / 3)
            ) / (768 * k * (-16 + k^2) * (-9 + k^2) * (-4 + k^2) * (-1 + k^2) * pi)
    end

    if k != 0
        fc /= sqrt(2)
    end

    return fc / norm_b4_cheb
end

Terms = Dict()
Terms[1] = [1, 5]
Terms[2] = [2, 6]
Terms[3] = [3, 7]
Terms[4] = [4, 8]

function f(x::Vector{Float64})::Float64
    if length(x) != 8
        error("Vector has to be 8-dimensional")
    end

    y = 0.0

    for i = 1:4
        y += B2(x[Terms[i][1]]) * B4(x[Terms[i][2]])
    end

    return y
end

function fc(k::Vector{Int64})::Float64
    if length(k) != 8
        error("Index has to be 8-dimensional")
    end

    ind = map(ki -> (ki == 0) ? 0 : 1, k)

    Terms_Support = Dict()

    for i = 1:4
        Terms_Support[i] = (sum(ind) == sum(ind[Terms[i]]))
    end

    y = 0.0

    for i = 1:4
        y += Terms_Support[i] * B2_chat(k[Terms[i][1]]) * B4_chat(k[Terms[i][2]])
    end

    return y
end

AS = Vector{Vector{Int64}}(undef, 13)
AS[1] = []
AS[2] = [1]
AS[3] = [2]
AS[4] = [3]
AS[5] = [4]
AS[6] = [5]
AS[7] = [6]
AS[8] = [7]
AS[9] = [8]
AS[10] = [1, 5]
AS[11] = [2, 6]
AS[12] = [3, 7]
AS[13] = [4, 8]

function norm()::Float64
    return sqrt(4 + 12 * B2_chat(0) * B2_chat(0) * B4_chat(0) * B4_chat(0))
end

function inverseDistribution(x::Float64)::Float64
    return sin(pi * (x - 0.5))
end

function generateData(M::Int)::Tuple{Array{Float64,2},Array{Float64,1}}
    X = 2 .* rand(8, M) .- 1
    X = inverseDistribution.(X)

    y = [f(X[:, i]) for i = 1:M]
    return (X, y)
end

end
