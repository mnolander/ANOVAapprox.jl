function get_l2error(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    return norm(y_eval - a.y) / norm(a.y)
end

function get_l2error(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    y_eval = evaluate(a, X, λ)
    return norm(y_eval - y) / norm(y)
end

function get_l2error(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_l2error(a, λ) for λ in collect(keys(a.fc)))
end

function get_l2error(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_l2error(a, X, y, λ) for λ in collect(keys(a.fc)))
end

function get_mse(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    return 1 / length(a.y) * (norm(y_eval - a.y)^2)
end

function get_mse(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    y_eval = evaluate(a, X, λ)
    return 1 / length(y) * (norm(y_eval - y)^2)
end

function get_mse(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_mse(a, λ) for λ in collect(keys(a.fc)))
end

function get_mse(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_mse(a, X, y, λ) for λ in collect(keys(a.fc)))
end

function get_mad(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    return 1 / length(a.y) * norm(y_eval - a.y, 1)
end

function get_mad(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    y_eval = evaluate(a, X, λ)
    return 1 / length(y) * norm(y_eval - y, 1)
end

function get_mad(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_mad(a, λ) for λ in collect(keys(a.fc)))
end

function get_mad(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_mad(a, X, y, λ) for λ in collect(keys(a.fc)))
end

function get_L2error(a::approx, norm::Float64, bc_fun::Function, λ::Float64)::Float64
    error = norm^2
    index_set = get_IndexSet(a.trafo.setting, size(a.X, 1))

    for i = 1:size(index_set, 2)
        k = index_set[:, i]
        error += abs(bc_fun(k) - a.fc[λ][i])^2 - abs(bc_fun(k))^2
    end

    return sqrt(error) / norm
end

function get_L2error(a::approx, norm::Float64, bc_fun::Function)::Dict{Float64,Float64}
    return Dict(λ => get_L2error(a, norm, bc_fun, λ) for λ in collect(keys(a.fc)))
end
