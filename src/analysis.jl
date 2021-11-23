@doc raw"""
    get_GSI( a::approx, λ::Float64; dict::Bool = false )::Union{Vector{Float64},Dict{Vector{Int},Float64}}

This function returns the global sensitivity indices of the approximation with ``\lambda`` as a vector for `dict = false` or else a dictionary.
"""
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

@doc raw"""
    get_GSI( a::approx; dict::Bool = false )::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}

This function returns the global sensitivity indices of the approximation for all ``\lambda`` as a vector for `dict = false` or else a dictionary.
"""
function get_GSI(
    a::approx;
    dict::Bool = false,
)::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}
    return Dict(λ => get_GSI(a, λ, dict = dict) for λ in collect(keys(a.fc)))
end

@doc raw"""
    get_AttributeRanking( a::approx, λ::Float64 )::Vector{Float64}

This function returns the attribute ranking of the approximation for reg. parameter ``\lambda`` as a vector of length `a.d`.
"""
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
        weights = 0.0
        for s in u
            r[s] += gsis[u] * 1/factors[i, s]
            weights += 1/factors[i, s]
        end
        nf += weights * gsis[u]
    end

    return r ./ nf
end

@doc raw"""
    get_AttributeRanking( a::approx, λ::Float64 )::Dict{Float64,Vector{Float64}}

This function returns the attribute ranking of the approximation for all reg. parameters ``\lambda`` as a dictionary of vectors of length `a.d`.
"""
function get_AttributeRanking(a::approx)::Dict{Float64,Vector{Float64}}
    return Dict(λ => get_AttributeRanking(a, λ) for λ in collect(keys(a.fc)))
end
