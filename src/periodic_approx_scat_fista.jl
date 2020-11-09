mutable struct periodic_approx_scat_fista{d,ds} <: periodic_approx 
  X::Matrix{Float64}
  y::Vector{ComplexF64}
  U::Vector{Vector{Int}}
  trafo::GroupedTransform
  N::Vector{Int}
  fc::Dict{Float64, GroupedCoeff}

  function periodic_approx_scat_fista{d,ds}( X::Matrix{Float64}, y::Vector{ComplexF64}, U::Vector{Vector{Int}}, trafo::GroupedTransform, N::Vector{Int} ) where {d,ds}
    return new(X, y, U, trafo, N, Dict{Float64, GroupedCoeff}())
  end 
end



function approximate( approx::periodic_approx_scat_fista{d,ds}, lambda::Float64; smoothness::Float64 = 2.0, max_iter::Int=25) where {d,ds}
  println("computing the approximation for λ = ", lambda)
  N = sum( s -> prod(s[:bandwidths].-1), approx.trafo.setting)

  λs = collect(keys(approx.fc)) # use the solution for the closest λ as an initial guess
  if length(λs) == 0
    ghat = GroupedCoeff(approx.trafo.setting, zeros(ComplexF64, N))
  else
    idx = argmin(λs .- lambda)
    ghat = GroupedCoeff(approx.trafo.setting, copy(approx.fc[λs[idx]].data))
  end
  
  what = sobolev_weights(approx.trafo.setting, smoothness = smoothness)
  fista!(ghat, approx.trafo, approx.y, lambda, what, max_iter = max_iter)
  approx.fc[lambda] = ghat
end



function approximate( approx::periodic_approx_scat_fista{d,ds}; lambda::Vector{Float64} = exp.(range(0, 5, length = 5)), args...) where {d,ds}
  sort!(lambda, lt = !isless) # biggest λ will be computed first such that the initial guess 0 is somewhat good
  for λ in lambda
    approximate(approx, λ; args...)
  end
end
