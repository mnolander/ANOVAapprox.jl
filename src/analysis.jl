function get_GSI(
    a::approx,
    λ::Float64;
    dict::Bool = false,
)::Union{Vector{Float64},Dict{Vector{Int},Float64}}
    variances = norms(a.fc[λ]) .^ 2
    variances = variances[2:end]
    variance_f = sum(variances)

    if dict
        gsis = Dict{Vector{Int},Float64}()
        for i = 1:length(a.fc[λ].setting)
            s = a.fc[λ].setting[i]
            u = s[:u]
            if u != []
                gsis[u] = norm(a.fc[λ][u])^2 / variance_f
            end
        end
        return gsis
    else
        return variances ./ variance_f
    end
end

function get_GSI(
    a::approx;
    dict::Bool = false,
)::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}
    return Dict(λ => get_GSI(a, λ, dict = dict) for λ in collect(keys(a.fc)))
end

function get_AttributeRanking(a::approx, λ::Float64)::Vector{Float64}
    d = size(a.X, 1)
    gsis = get_GSI(a, λ, dict = true)
    U = collect(keys(gsis))
    factors = zeros(Int64, length(U), d)

    for i = 1:length(U)
        u = U[i]
        for s in u
            for v in U
                if (length(u) == length(v)) && (s in v)
                    factors[i, s] += 1
                end
            end
        end
    end

    r = zeros(Float64, d)
    nf = 0.0

    for i = 1:length(U)
        u = U[i]
        for s in u
            r[i] += gsis[u] * factors[i, s]
        end
        nf += gsis[u] * sum(factors[i, :])
    end

    return r ./ nf
end

function get_AttributeRanking(a::approx)::Dict{Float64,Vector{Float64}}
    return Dict(λ => get_AttributeRanking(a, λ) for λ in collect(keys(a.fc)))
end
