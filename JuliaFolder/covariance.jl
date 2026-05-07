using LoopVectorization
using LinearAlgebra


using Statistics
using JLD2


function get_covariance(M::Matrix{Float64}, lag::Matrix{Integer}, m_size::Integer = 32)
    local Σ::Matrix{Float64} = zeros(m_size, m_size)
    @inbounds for i ∈ 1:m_size
        for j ∈ 1:m_size
            offset = lag[i, j]
            if offset == 0
                Σ[i, j] = cov(M[:, i], M[:, j])
            elseif offset > 0
                Σ[i, j] = cov(M[1 + offset:end, i], M[1:end - offset, j])
            elseif offset < 0
                Σ[i, j] = cov(M[1:end + offset, i], M[1 - offset:end, j])
            end
        end
    end
    return Σ
end


#=
function get_correlation(M::Matrix{Float64}, lag::Matrix{Integer}, m_size::Integer = 32)
    local Σ::Matrix{Float64} = zeros(m_size, m_size)
    @inbounds for i ∈ 1:m_size
        for j ∈ 1:m_size
            offset = lag[i, j]
            if offset == 0
                Σ[i, j] = cor(M[:, i], M[:, j])
            elseif offset > 0
                Σ[i, j] = cor(M[1 + offset:end, i], M[1:end - offset, j])
            elseif offset < 0
                Σ[i, j] = cor(M[1:end + offset, i], M[1 - offset:end, j])
            end
        end
    end
    return Σ
end
=#


function nearest_psd(Σ::Matrix{Float64})
    Σ_symmetric::Matrix{Float64} = UpperTriangular(Σ)' + UpperTriangular(Σ) - Diagonal(Σ)
    decomposition = eigen(Σ_symmetric)
    λ_vec::Vector{Float64} = decomposition.values
    v_mat::Matrix{Float64} = decomposition.vectors
    return Symmetric(v_mat * Diagonal(max.(0.0, λ_vec)) * v_mat^-1)
end


function covariance_to_correlation(Σ::Symmetric{Float64, Matrix{Float64}})
    diag_factor::Matrix{Float64} = sqrt.(Diagonal(Σ))^-1
    return Symmetric(diag_factor * Σ * diag_factor)
end


function apply_threshold(Σ::Symmetric{Float64, Matrix{Float64}}, R::Symmetric{Float64, Matrix{Float64}}, threshold::Float64)
    R_values::Vector{Float64} = eigen(R).values
    decomposition = eigen(Σ)
    Σ_values::Vector{Float64} = decomposition.values
    Σ_vectors::Matrix{Float64} = decomposition.vectors
    return Symmetric(Σ_vectors * Diagonal(map(i -> if R_values[i] > threshold return Σ_values[i] else return 0.0 end, 1:32)) * Σ_vectors^-1)
end


t_samples::Integer = 1792
#normalization_factor::Integer = 2 * t_samples^2
threshold::Float64 = (1.0 + sqrt(32 / 1770))^2


sdat = load(normpath(joinpath((@__FILE__), raw"..\subject5dat.jld2")))
lag_matrix::Matrix{Integer} = sdat["lag_matrix"]
#test_correlation = covariance_to_correlation(nearest_psd(get_covariance(sdat["CISI_data"][1], lag_matrix)))
#test_covariance = nearest_psd(get_covariance(sdat["CISI_data"][1], lag_matrix))
#apply_threshold(test_correlation, test_covariance, threshold)
#apply_threshold(nearest_psd(get_covariance(sdat["CISI_data"][1], lag_matrix)), covariance_to_correlation(nearest_psd(get_covariance(sdat["CISI_data"][1], lag_matrix))), threshold)
oldcov = nearest_psd(get_covariance(sdat["VISI_data"][1], lag_matrix))
oldcor = covariance_to_correlation(oldcov)
old_correlation = apply_threshold(oldcov, oldcor, threshold)
new_correlation = apply_threshold(oldcov, oldcor, threshold)
norm_change = zeros(119)
for i ∈ 2:120
    newcov = nearest_psd(get_covariance(sdat["CISI_data"][i], lag_matrix))
    newcor = covariance_to_correlation(newcov)
    new_correlation = apply_threshold(newcov, newcor, threshold)
    norm_change[i - 1] = norm(new_correlation - old_correlation)
    old_correlation = new_correlation
end


#=
Figure for plotting the change
using GLMakie
fig = Figure(size = (1600, 900))
ax = Axis(
    fig[1, 1], title = "Processed Covariance Matrix Change Between Trials", xlabel = "Trial Number", ylabel = "p2-norm of Matrix-Valued Difference Between Trials",
    xticks = [20; 40; 60; 80; 100]
)
lines!(ax, 1:119, norm_change)
xlims!(ax, 1, 119)
save(normpath(joinpath((@__FILE__), raw"..\normchange.png")), fig)
fig
=#



#=
temp_cov = get_covariance(matrix_1, lag_matrix, 32)
#temp_cor = get_correlation(matrix_1, lag_matrix, 32)
test_cov = UpperTriangular(temp_cov)' + UpperTriangular(temp_cov) - Diagonal(temp_cov)
test_cor = UpperTriangular(temp_cor)' + UpperTriangular(temp_cor) - Diagonal(temp_cor)
cov_eig = eigen(test_cov)
cov_eigval = cov_eig.values
cov_eigvec = cov_eig.vectors
test_cov′ = Symmetric(cov_eigvec * Diagonal(max.(0.0, cov_eigval)) * cov_eigvec^-1)


diag_factor = sqrt.(Diagonal(test_cov′))^-1
test_cor′ = Symmetric(diag_factor * test_cov′ * diag_factor)
cor_eigval′ = eigen(test_cor′).values
nonsig_indices = findall(!>(threshold), cor_eigval′)
cov_eigval[nonsig_indices] = zeros(length(nonsig_indices))
test_cov′ = Symmetric(cov_eigvec * Diagonal(cov_eigval) * cov_eigvec^-1)
diag_factor = sqrt.(Diagonal(test_cov′))^-1
test_cor′′ = Symmetric(diag_factor * test_cov′ * diag_factor)
=#

#eig_true = cov_eig.vectors * Diagonal()
#test_eig = eigen(test_cov)
#test_eig.vectors * Diagonal(max.(0.0, test_eig.values)) * test_eig.vectors^-1


