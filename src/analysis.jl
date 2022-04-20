@doc raw"""
    get_GSI( a::approx, λ::Float64; dict::Bool = false )::Union{Vector{Float64},Dict{Vector{Int},Float64}}

This function returns the global sensitivity indices of the approximation with ``\lambda`` as a vector for `dict = false` or else a dictionary.
"""
function get_GSI(
    a::approx,
    λ::Float64;
    dict::Bool = false,
)#::Union{Vector{Float64},Dict{Vector{Int},Float64}}
    if a.basis == "wav1"
        variances = norms(a.fc[λ],1,dict=false) .^ 2
    elseif a.basis == "wav2"
        variances = norms(a.fc[λ],2,dict=false) .^ 2
    elseif a.basis == "wav3"
        variances = norms(a.fc[λ],3,dict=false) .^ 2
    elseif a.basis == "wav4"
        variances = norms(a.fc[λ],4,dict=false) .^ 2
    else
        variances = norms(a.fc[λ]) .^ 2
    end
    variances = variances[2:end]
    variance_f = sum(variances)

    if dict
        gsis = Dict{Vector{Int},Float64}()
        if a.basis == "wav1"
            variances = norms(a.fc[λ],1,dict=true)
        elseif a.basis == "wav2"
            variances = norms(a.fc[λ],2,dict=true)
        elseif a.basis == "wav3"
            variances = norms(a.fc[λ],3,dict=true)
        elseif a.basis == "wav4"
            variances = norms(a.fc[λ],4,dict=true)
        else
            variances = norms(a.fc[λ],dict = true)
        end

        return Dict{Vector{Int},Float64}(Dict((u,variances[u]^2/variance_f) for u in keys(variances)))

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
)#::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}
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
    lengths = [ length(u) for u in U ]
    ds = maximum(lengths)

    factors = zeros(Int64, d, ds)

    for i = 1:d
        for j = 1:ds
            for v in U
                if (i in v) && (length(v) == j)
                    factors[i,j] += 1
                end
            end
        end
    end

    r = zeros(Float64, d)
    nf = 0.0

    for u in U
        weights = 0.0
        for s in u
            r[s] += gsis[u] * 1/factors[s, length(u)]
            weights += 1/factors[s, length(u)]
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

function get_ActiveSet( a::approx, eps::Vector{Float64}, λ::Float64 )::Vector{Vector{Int}}
    U = a.U[2:end]
    lengths = [ length(u) for u in U ]
    ds = maximum(lengths)

    if length(eps) != ds
        error( "Entries in vector eps have to be ds.")
    end

    gsi = get_GSI(a, λ)

    n = 0

    for i = 1:length(gsi)
        if gsi[i] > eps[length(U[i])]
            n += 1
        end
    end

    U_active = Vector{Vector{Int}}(undef, n+1)
    U_active[1] = []
    idx = 2

    for i = 1:length(gsi)
        if gsi[i] > eps[length(U[i])]
            U_active[idx] = U[i]
            idx += 1
        end
    end

    return U_active
end

function get_ActiveSet(a::approx, eps::Vector{Float64})::Dict{Float64,Vector{Vector{Int}}}
    return Dict(λ => get_ActiveSet(a, eps, λ) for λ in collect(keys(a.fc)))
end
