#----- pkg

using Serialization
using JLD2
using NPZ
using LinearAlgebra
using ProgressMeter
using LoopVectorization

###

function g_AB(A::Int64,B::Int64,c6::Float64,r_idx::Int64,Q_idx::Int64,grid_size::String,hD::Float64,model::String,metric_terms::Vector{<:Function},out::String,output_format::String)::Float64

    #----- prepare fields

    d1 = npzread("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d1U_r=$(r_idx)_Q=$(Q_idx).npy")
    d2 = npzread("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d2U_r=$(r_idx)_Q=$(Q_idx).npy")
    d3 = npzread("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/d3U_r=$(r_idx)_Q=$(Q_idx).npy")

    l1 = length(d1[:,1,1,1]); l2 = length(d1[1,:,1,1]); l3 = length(d1[1,1,:,1])

    #----- main

    println()
    println("#--------------------------------------------------#")
    println()
    println("Metric (A=$(A), B=$(B)) --- r_idx=$(r_idx), Q_idx=$(Q_idx)")
    println()
 
    DA = npzread("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(A)U_r=$(r_idx)_Q=$(Q_idx).npy")
    DB = npzread("/home/velni/phd/w/nnp/data/deriv/$(model)/$(grid_size)/D$(B)U_r=$(r_idx)_Q=$(Q_idx).npy")

    g = 0.

    if model == "std"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]

        @showprogress 1 "Computing..." for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3))
            end
        end

    elseif model == "gen"
        g1 = metric_terms[1]
        g2 = metric_terms[2]
        g3 = metric_terms[3]
        g4 = metric_terms[4]
        g5 = metric_terms[5]
        g6 = metric_terms[6]
        g7 = metric_terms[7]

        @showprogress 1 "Computing" for k in 1:l3
            @inbounds @fastmath for j in 1:l2, i in 1:l1
                G1 = g1(i,j,k,DA,DB,d1,d2,d3)
                G2 = g2(i,j,k,DA,DB,d1,d2,d3)
                G3 = g3(i,j,k,DA,DB,d1,d2,d3)
                G4 = g4(i,j,k,DA,DB,d1,d2,d3)
                G5 = g5(i,j,k,DA,DB,d1,d2,d3)
                G6 = g6(i,j,k,DA,DB,d1,d2,d3)
                G7 = g7(i,j,k,DA,DB,d1,d2,d3)

                g = g + (2*(G1 + G2 - G3 + c6*G4 - c6*G5 + (c6/2.)*G6 - (c6/2.)*G7) )
            end
        end
    end

    return g*hD^3
end
