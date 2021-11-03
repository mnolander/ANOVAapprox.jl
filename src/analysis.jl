function get_GSI(
    a::approx,
    lambda::Float64;
    dict::Bool = false,
)::Union{Vector{Float64},Dict{Vector{Int},Float64}}
    variances = norms(a.fc[lambda]) .^ 2
    variances = variances[2:end]
    variance_f = sum(variances)

    if dict
        gsis = Dict()
        for i = 1:length(a.fc[lambda].setting)
            s = a.fc[lambda].setting[i]
            u = s[:u]
            if u != []
                gsis[u] = norm(a.fc[lambda][u])^2 / variance_f
            end
        end
        return gsis
    else
        return variances ./ variance_f
    end
end
