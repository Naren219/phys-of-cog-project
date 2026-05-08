using LinearAlgebra
using ProximalOperators
using CSV
using DataFrames


dat1 = CSV.read(normpath(joinpath(@__FILE__, raw"..\data2.csv")), DataFrame)
dat2 = CSV.read(normpath(joinpath(@__FILE__, raw"..\lag_samples_matrix.csv")), DataFrame)


cor::Matrix{Float64} = Matrix{Float64}(dat1[:, :])
samples::Matrix{Integer} = Matrix{Integer}(dat2[:, :])


decomp = eigen(cor)
cor_diag::Matrix{Float64} = Diagonal(max.(0.0, decomp.values))
cor_p::Matrix{Float64} = decomp.vectors
nearest_psd_frobenius::Matrix{Float64}, __ = prox(IndPSD(), cor + 10.0^-13 * I)# + 10.0^-14 * I)
threshold::Float64 = (1.0 + sqrt(32 / minimum(samples)))^2
big_eigenvalues::Vector{Float64} = filter(x -> x > threshold, eigen(nearest_psd_frobenius).values)
my_eigenvalues::Vector{Float64} = zeros(32)
my_eigenvalues[end - length(big_eigenvalues) + 1 : end] = big_eigenvalues
big_cor_matrix::Matrix{Float64} = cor_p * Diagonal(my_eigenvalues) * cor_p^-1


