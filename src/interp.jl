#----- pkg

using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions
using ProgressMeter

#----- interp

using DelimitedFiles
using Interpolations

function interp_2sky_no(rtc,r_vals::Array{Float64}, model::String,data, out::String,output_format::String)

    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = length(y3)

    #----- main loop
    
    r0_itp = first(r0):0.01:last(r0)
    itp = extrapolate(scale(interpolate(f0,BSpline(Linear())), r0_itp), Line())

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation")
     
    f_plus = Array{Float64,3}[]; f_minus = Array{Float64,3}[]
    for r_idx in 1:length(r_vals)
        
        matrix_f_plus = zeros(Float64, l1,l2,l3); matrix_f_minus = zeros(Float64, l1,l2,l3)

        println()

        r = r_vals[r_idx]
        
        @showprogress 1 "Computing r=$(r_idx):" for k in 1:length(y3)
            @inbounds @fastmath for j in 1:length(y2)
                for i in 1:length(y1)
                    temp_f_plus = itp(norm([y1[i],y2[j],y3[k]] .+ [0.,0.,r/2]) ) 
                    temp_f_minus = itp(norm([y1[i],y2[j],y3[k]] .- [0.,0.,r/2]) )
            
                    matrix_f_plus[i,j,k] = temp_f_plus
                    matrix_f_minus[i,j,k] = temp_f_minus
                end
            end
        end 

        #----- data saving

        if output_format == "jld2"
            path1 = out*"/f_$(model)_plus_r=$(r_idx).jld2"; path2 = out*"/f_$(model)_minus_r=$(r_idx).jld2"
            @save path1 matrix_f_plus; @save path2 matrix_f_minus
        elseif output_format == "npy"
            npzwrite(out*"/f_$(model)_plus_r=$(r_idx).npy", matrix_f_plus); npzwrite(out*"/f_$(model)_minus_r=$(r_idx).npy", matrix_f_minus)
        end

    end
        
    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

#####

function interp_2sky_dx(rtc,r_vals, model::String,deriv::String,hD::Float64, out::String,output_format::String)

    data = readdlm("/home/velni/phd/w/nnp/data/profile_f/profile_f_$(model).txt")
    r0 = data[:,1]; f0 = data[:,2]

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]
    l1 = length(y1); l2 = length(y2); l3 = length(y3)

    idx_list = []
    for i in 1:length(y1)
        for j in 1:length(y2)
            for k in 1:length(y3)
                push!(idx_list, [i,j,k])
            end
        end
    end

    #----- main loop
    
    r0_itp = first(r0):0.01:last(r0)
    itp = extrapolate(scale(interpolate(f0,BSpline(Linear())), r0_itp), Line())

    println()
    println("#--------------------------------------------------#")
    println()
    println("Radial function interpolation")
    println()
    
    if deriv == "x1"
        x_p = [hD,0.,r]./2; x_m = [-hD,0.,r]./2
    elseif deriv == "x2"
        x_p = [0.,hD,r]./2; x_m = [0.,-hD,r]./2
    elseif deriv == "x3"
        x_p = [0.,0.,hD+r]./2; x_m = [0.,0.,-hD+r]./2
    end

    f_plus_p = Matrix{Float64}[]; f_plus_m = Matrix{Float64}[]; f_minus_p = Matrix{Float64}[]; f_minus_m = Matrix{Float64}[]
    for (r_idx,r) in enumerate(r_vals)
        matrix_f_plus_p = zeros(Float64, l1,l2,l3); matrix_f_plus_m = zeros(Float64, l1,l2,l3); matrix_f_minus_p = zeros(Float64, l1,l2,l3); matrix_f_minus_m = zeros(Float64, l1,l2,l3)

        print("computing: r=$(r_idx)")
        println()
        
        @showprogress 1 "Computing..." for idx in idx_list
            temp_f_plus_p = itp(norm([y1[idx[1]],y2[idx[2]],y3[idx[3]]] .+ x_p) ) 
            temp_f_plus_m = itp(norm([y1[idx[1]],y2[idx[2]],y3[idx[3]]] .+ x_m) )
            temp_f_minus_p = itp(norm([y1[idx[1]],y2[idx[2]],y3[idx[3]]] .- x_p) )
            temp_f_minus_m = itp(norm([y1[idx[1]],y2[idx[2]],y3[idx[3]]] .- x_m) )

            matrix_f_plus_p[idx[1],idx[2],idx[3]] = temp_f_plus_p
            matrix_f_plus_m[idx[1],idx[2],idx[3]] = temp_f_plus_m
            matrix_f_minus_p[idx[1],idx[2],idx[3]] = temp_f_minus_p
            matrix_f_minus_m[idx[1],idx[2],idx[3]] = temp_f_minus_m
        end
        
        println()
        push!(f_plus_p, matrix_f_plus_p)
        push!(f_plus_m, matrix_f_plus_m)
        push!(f_minus_p, matrix_f_minus_p)
        push!(f_minus_m, matrix_f_minus_m)
    end

    #----- data saving

    if output_format == "jld2"
        path1 = out*"/f_$(model)_$(deriv)_plus_p.jld2"; path2 = out*"/f_$(model)_$(deriv)_plus_m.jld2"
        path3 = out*"/f_$(model)_$(deriv)_minus_p.jld2"; path4 = out*"/f_$(model)_$(deriv)_minus_m.jld2"
        @save path1 f_plus_p; @save path2 f_plus_m; @save path3 f_minus_p; @save path4 f_minus_m; 

    elseif output_format == "npy"
        npzwrite(out*"/f_$(model)_$(deriv)_plus_p.npy", cat(f_plus_p...;dim=4)); npzwrite(out*"/f_$(model)_$(deriv)_plus_m.npy", cat(f_plus_m...;dim=4))
        npzwrite(out*"/f_$(model)_$(deriv)_minus_p.npy", cat(f_minus_p...;dim=4)); npzwrite(out*"/f_$(model)_$(deriv)_minus_m.npy", cat(f_minus_m...;dim=4))
    
    end
    
    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")
end

