abstract type approx end

mutable struct approx_ds <: approx
    basis::String
    X::Matrix{Float64}
    y::Union{Vector{ComplexF64},Vector{Float64}}
    ds::Int
    N::Vector{Int}
    trafo::GroupedTransform
    fc::Dict{Float64,GroupedCoefficients}

    function approx_ds(
        X::Matrix{Float64},
        y::Union{Vector{ComplexF64},Vector{Float64}},
        ds::Int,
        N::Vector{Int},
        basis::String = "cos",
    )
        if basis in bases
            d = size(X, 1)
            M = size(X, 2)

            if !isa(y, vtypes[basis])
                error(
                    "Periodic functions require complex vectors, nonperiodic functions real vectors.",
                )
            end

            if length(y) != M
                error("y needs as many entries as X has columns.")
            end

            if (ds < 1) || (ds > d)
                error("ds needs to be between 1 and d.")
            end

            if length(N) != ds
                error("N needs to have ds entries.")
            end

            if (basis == "per") && ((minimum(X) < -0.5) || (maximum(X) >= 0.5))
                error("Nodes need to be between -0.5 and 0.5.")
            elseif (basis == "cos") && ((minimum(X) < 0) || (maximum(X) > 1))
                error("Nodes need to be between 0 and 1.")
            elseif (basis == "cheb") && ((minimum(X) < -1) || (maximum(X) > 1))
                error("Nodes need to be between -1 and 1.")
            end

            Xt = copy(X)

            if basis == "cos"
                Xt ./= 2
            elseif basis == "cheb"
                Xt = acos.(Xt)
                Xt ./= 2 * pi
            end

            trafo = GroupedTransform(gt_systems[basis], d, ds, N, Xt)
            return new(basis, X, y, ds, N, trafo, Dict{Float64,GroupedCoefficients}())
        else
            error("Basis not found.")
        end
    end
end

mutable struct approx_U <: approx
    basis::String
    X::Matrix{Float64}
    y::Union{Vector{ComplexF64},Vector{Float64}}
    U::Vector{Vector{Int}}
    N::Vector{Int}
    trafo::GroupedTransform
    fc::Dict{Float64,GroupedCoefficients}

    function approx_U(
        X::Matrix{Float64},
        y::Union{Vector{ComplexF64},Vector{Float64}},
        U::Vector{Vector{Int}},
        N::Vector{Int},
        basis::String = "cos",
    )
        if basis in bases
            M = size(X, 2)

            if !isa(y, vtypes[basis])
                error(
                    "Periodic functions require complex vectors, nonperiodic functions real vectors.",
                )
            end

            if length(y) != M
                error("y needs as many entries as X has columns.")
            end

            if length(N) != length(U)
                error("N needs to have |U| entries.")
            end

            if (basis == "per") && ((minimum(X) < -0.5) || (maximum(X) >= 0.5))
                error("Nodes need to be between -0.5 and 0.5.")
            elseif (basis == "cos") && ((minimum(X) < 0) || (maximum(X) > 1))
                error("Nodes need to be between 0 and 1.")
            elseif (basis == "cheb") && ((minimum(X) < -1) || (maximum(X) > 1))
                error("Nodes need to be between -1 and 1.")
            end

            Xt = copy(X)

            if basis == "cos"
                Xt ./= 2
            elseif basis == "cheb"
                Xt = acos.(Xt)
                Xt ./= 2 * pi
            end

            trafo = GroupedTransform(gt_systems[basis], U, N, Xt)
            return new(basis, X, y, U, N, trafo, Dict{Float64,GroupedCoefficients}())
        else
            error("Basis not found.")
        end
    end
end

function approximate(
    a::approx,
    λ::Float64;
    max_iter::Int = 50,
    weights::Union{Vector{Float64},Nothing} = nothing,
    verbose::Bool = false,
    solver::String = "lsqr",
)::Nothing
    M = size(a.X, 2)
    nf = get_NumFreq(a.trafo.setting)

    w = ones(Float64, nf)

    if !isnothing(weights)
        if (length(weights) != nf) || (minimum(weights) < 1)
            error("Weight requirements not fulfilled.")
        else
            w = weights
        end
    end

    if a.basis == "per"
        what = GroupedCoefficients(a.trafo.setting, complex(w))
    else
        what = GroupedCoefficients(a.trafo.setting, w)
    end

    λs = collect(keys(a.fc))
    tmp = zeros(types[a.basis], nf)

    if length(λs) != 0
        idx = argmin(λs .- λ)
        tmp = copy(a.fc[λs[idx]].data)
    end

    if solver == "lsqr"
        diag_w_sqrt = sqrt(λ) .* w
        if a.basis == "per"
            F_vec = LinearMap{ComplexF64}(
                fhat -> vcat(
                    a.trafo * GroupedCoefficients(a.trafo.setting, fhat),
                    diag_w_sqrt .* fhat,
                ),
                f -> vec(a.trafo' * f[1:M]) + diag_w_sqrt .* f[M+1:end],
                M + nf,
                nf,
            )
            lsqr!(
                tmp,
                F_vec,
                vcat(a.y, zeros(ComplexF64, nf)),
                maxiter = max_iter,
                verbose = verbose,
            )
            a.fc[λ] = GroupedCoefficients(a.trafo.setting, tmp)
        else
            F_vec = LinearMap{Float64}(
                fhat -> vcat(
                    a.trafo * GroupedCoefficients(a.trafo.setting, fhat),
                    diag_w_sqrt .* fhat,
                ),
                f -> vec(a.trafo' * f[1:M]) + diag_w_sqrt .* f[M+1:end],
                M + nf,
                nf,
            )
            lsqr!(
                tmp,
                F_vec,
                vcat(a.y, zeros(Float64, nf)),
                maxiter = max_iter,
                verbose = verbose,
            )
            a.fc[λ] = GroupedCoefficients(a.trafo.setting, tmp)
        end
    elseif solver == "fista"
        ghat = GroupedCoefficients(a.trafo.setting, tmp)
        fista!(ghat, a.trafo, a.y, λ, what, max_iter = max_iter)
        a.fc[λ] = ghat
    else
        error("Solver not found.")
    end

    return
end


function approximate(
    a::approx;
    lambda::Vector{Float64} = exp.(range(0, 5, length = 5)),
    args...,
)::Nothing
    sort!(lambda, lt = !isless) # biggest λ will be computed first such that the initial guess 0 is somewhat good
    for λ in lambda
        approximate(a, λ; args...)
    end
    return
end


function evaluate(
    a::approx,
    X::Matrix{Float64},
    λ::Float64,
)::Union{Vector{ComplexF64},Vector{Float64}}
    basis = a.basis

    if (basis == "per") && ((minimum(X) < -0.5) || (maximum(X) >= 0.5))
        error("Nodes need to be between -0.5 and 0.5.")
    elseif (basis == "cos") && ((minimum(X) < 0) || (maximum(X) > 1))
        error("Nodes need to be between 0 and 1.")
    elseif (basis == "cheb") && ((minimum(X) < -1) || (maximum(X) > 1))
        error("Nodes need to be between -1 and 1.")
    end

    Xt = copy(X)

    if basis == "cos"
        Xt ./= 2
    elseif basis == "cheb"
        Xt = acos.(Xt)
        Xt ./= 2 * pi
    end

    if isa(a, approx_U)
        trafo = GroupedTransform(gt_systems[basis], a.U, a.N, Xt)
        return trafo * a.fc[λ]
    elseif isa(a, approx_ds)
        trafo = GroupedTransform(gt_systems[basis], size(X, 1), a.ds, a.N, Xt)
        return trafo * a.fc[λ]
    end
end

function evaluate(a::approx, λ::Float64)::Union{Vector{ComplexF64},Vector{Float64}}
    return a.trafo * a.fc[λ]
end

function evaluate(
    a::approx,
    X::Matrix{Float64},
)::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}
    return Dict(λ => evaluate(a, X, λ) for λ in collect(keys(a.fc)))
end

function evaluate(a::approx)::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}
    return Dict(λ => evaluate(a, λ) for λ in collect(keys(a.fc)))
end
