using JLD2
using CSV
using DataFrames


CISI_data::Vector{Matrix{Float64}} = repeat([zeros(1792, 32)], 120)
VISI_data::Vector{Matrix{Float64}} = repeat([zeros(1792, 32)], 120)
for i ∈ 1:120
    CISI_data[i] = Matrix{Float64}(CSV.read(normpath(joinpath((@__FILE__), "..\\CISI_data\\CISI$i.csv")), DataFrame))
    VISI_data[i] = Matrix{Float64}(CSV.read(normpath(joinpath((@__FILE__), "..\\VISI_data\\VISI$i.csv")), DataFrame))
end
lag_matrix::Matrix{Integer} = Matrix{Float64}(CSV.read(normpath(joinpath((@__FILE__), raw"..\lagamount.csv")), DataFrame))


# i, j
# j is lagged one
# normal matrix convention
# i is row number, j is column number
# sample frequency is 512 Hz


save(normpath(joinpath((@__FILE__), raw"..\subject5dat.jld2")), Dict(
    "CISI_data" => CISI_data,
    "VISI_data" => VISI_data,
    "lag_matrix" => lag_matrix,
))

