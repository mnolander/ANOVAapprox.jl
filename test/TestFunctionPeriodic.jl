module TestFunctionPeriodic

using LinearAlgebra

C = zeros(Float64, 3)
C[1] = sqrt(0.75)
C[2] = sqrt(315 / 604)
C[3] = sqrt(277200 / 655177)

# B_2(x) for x in [0,1)
function b_spline_2(x::Float64)::Float64
    C_2 = C[1]

    if (x >= 0.0) && (x < 0.5)
        return C_2 * 4 * x
    elseif (x >= 0.5) && (x < 1.0)
        return C_2 * 4 * (1 - x)
    else
        error("B-spline 2: out of range")
    end
end

# B_4(x) for x in [0,1)
function b_spline_4(x::Float64)::Float64
    C_4 = C[2]

    if (x >= 0.0) && (x < 0.25)
        return C_4 * 128 / 3 * x^3
    elseif (x >= 0.25) && (x < 0.5)
        return C_4 * (8 / 3 - 32 * x + 128 * x^2 - 128 * x^3)
    elseif (x >= 0.5) && (x < 0.75)
        return C_4 * (-88 / 3 - 256 * x^2 + 160 * x + 128 * x^3)
    elseif (x >= 0.75) && (x < 1.0)
        return C_4 * (128 / 3 - 128 * x + 128 * x^2 - (128 / 3) * x^3)
    else
        error("B-spline 4: out of range")
    end
end

# B_6(x) for x in [0,1)
function b_spline_6(x::Float64)::Float64
    C_6 = C[3]

    if (x >= 0.0) && (x < 1.0 / 6.0)
        return C_6 * 1944 / 5 .* x^5
    elseif (x >= 1.0 / 6.0) && (x < 2.0 / 6.0)
        return C_6 * (3 / 10 - 9 * x + 108 * x^2 - 648 * x^3 + 1944 * x^4 - 1944 * x^5)
    elseif (x >= 2.0 / 6.0) && (x < 0.5)
        return C_6 *
               (-237 / 10 + 351 * x - 2052 * x^2 + 5832 * x^3 - 7776 * x^4 + 3888 * x^5)
    elseif (x >= 0.5) && (x < 4.0 / 6.0)
        return C_6 *
               (2193 / 10 + 7668 * x^2 - 2079 * x + 11664 * x^4 - 13608 * x^3 - 3888 * x^5)
    elseif (x >= 4.0 / 6.0) && (x < 5.0 / 6.0)
        return C_6 *
               (-5487 / 10 + 3681 * x - 9612 * x^2 + 12312 * x^3 - 7776 * x^4 + 1944 * x^5)
    elseif (x >= 5.0 / 6.0) && (x < 1.0)
        return C_6 *
               (1944 / 5 - 1944 * x + 3888 * x^2 - 3888 * x^3 + 1944 * x^4 - 1944 / 5 * x^5)
    else
        error("B-spline 6: out of range")
    end
end

m1 = [1, 3]
m2 = [2, 5]
m3 = [4, 6]

AS = Vector{Vector{Int64}}(undef, 10)
AS[1] = []
AS[2] = [1]
AS[3] = [2]
AS[4] = [3]
AS[5] = [4]
AS[6] = [5]
AS[7] = [6]
AS[8] = [1, 3]
AS[9] = [2, 5]
AS[10] = [4, 6]

trans(x::Float64)::Float64 = x < 0 ? 1 + x : x

function f(x::Vector{Float64})::Float64
    if length(x) != 6
        error("Argument has to be 6-dimensional")
    end

    if !isempty(x[(x.>0.5).|(x.<-0.5)])
        error("The nodes have to be between -0.5 and 0.5.")
    end

    xT = trans.(x)

    return prod(b_spline_2.(xT[m1])) + prod(b_spline_4.(xT[m2])) + prod(b_spline_6.(xT[m3]))
end

sinc(x::Float64)::Float64 = (x == 0.0) ? 1 : sin(x) / x
b(k::Int64, r::Int64)::Float64 = C[Integer(r / 2)] * (sinc(pi * k / r))^r * cos(pi * k)

function fc(k::Vector{Int64})::Float64
    if length(k) != 6
        error("Index has to be 6-dimensional")
    end

    ind = map(ki -> (ki == 0) ? 0 : 1, k)

    b2_block = (sum(ind) == sum(ind[m1]))
    b4_block = (sum(ind) == sum(ind[m2]))
    b6_block = (sum(ind) == sum(ind[m3]))

    return b2_block * prod(map(ki -> b(ki, 2), k[m1])) +
           b4_block * prod(map(ki -> b(ki, 4), k[m2])) +
           b6_block * prod(map(ki -> b(ki, 6), k[m3]))
end

function norm()::Float64
    norm = 3.0

    for i = 1:2
        for j = i+1:3
            norm += 2 * (b(0, 2 * i))^2 * (b(0, 2 * j))^2
        end
    end

    return sqrt(norm)
end

end
