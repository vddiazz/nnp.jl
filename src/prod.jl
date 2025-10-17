#----- pkg

using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions

#----- field

function field_value(y1,y2,y3,r_vals,f_plus,f_minus, coord::Vector{Int64}, r_idx::Int, Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64})

    #----- params

    sigma = [[0im 1.; 1. 0im],
             [0. -1im; 1im 0.],
             [1. 0im; 0im -1.]]

    y = [y1[coord[1]], y2[coord[2]], y3[coord[3]]]
    xm = [0., 0., r_vals[r_idx]/2.]
    pos_p = y.+xm; pos_m = y.-xm

    ar_p = norm(pos_p, 2)
    ar_m = norm(pos_m, 2)

    #----- f(r) values

    f_p = f_plus[coord[1],coord[2],coord[3]][1]
    f_m = f_minus[coord[1],coord[2],coord[3]][1]

    #----- U1

    dirs1 = [(Q1*s)*inv(Q1) for s in sigma]
    Z1 = pos_m[1]*dirs1[1] + pos_m[2]*dirs1[2] + pos_m[3]*dirs1[3]
    dirs1_param = [0.5*tr(Z1), 0.5*tr(sigma[1]*Z1), 0.5*tr(sigma[2]*Z1), 0.5*tr(sigma[3]*Z1)]
    phi1 = [cos(f_m)+dirs1_param[1], sin(f_m)*(1/ar_m)*dirs1_param[2], sin(f_m)*(1/ar_m)*dirs1_param[3], sin(f_m)*(1/ar_m)*dirs1_param[4]]

    #----- U2

    dirs2 = [(Q2*s)*inv(Q2) for s in sigma]
    Z2 = pos_p[1]*dirs2[1] + pos_p[2]*dirs2[2] + pos_p[3]*dirs2[3]
    dirs2_param = [0.5*tr(Z2), 0.5*tr(sigma[1]*Z2), 0.5*tr(sigma[2]*Z2), 0.5*tr(sigma[3]*Z2)]
    phi2 = [cos(f_p)+dirs2_param[1], sin(f_p)*(1/ar_p)*dirs2_param[2], sin(f_p)*(1/ar_p)*dirs2_param[3], sin(f_p)*(1/ar_p)*dirs2_param[4]]

    #----- symmetrized product field

    U = [ phi1[1]*phi2[1] - phi1[2]*phi2[2] - phi1[3]*phi2[3] - phi1[4]*phi2[4],
          phi1[1]*phi2[2] + phi1[2]*phi2[1],
          phi1[1]*phi2[3] + phi1[3]*phi2[1],
          phi1[1]*phi2[4] + phi1[4]*phi2[1]]

    #----- normalization

    C0 = (U[1])^2; Ck = U[2]^2 + U[3]^2 + U[4]^2
    N = sqrt(C0 + Ck)

    #----- field
    
    return (1/N) .* U
end

###

function field_grid(r_vals, grid_size, r_idx::Int, Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare rtc and params

    y1 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y1.npy")
    y2 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y2.npy")
    y3 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y3.npy")

    idx_list = []
    for i in 1:length(y1)
        for j in 1:length(y2)
            for k in 1:length(y3)
                push!(idx_list, [i,j,k])
            end
        end
    end

    f_plus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_plus.npy")
    f_minus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_minus_r=61.npy")

    #----- evaluate field values
    
    U_vals = zeros(Float64, length(y1),length(y2),length(y3),4)

    for (step,idx) in enumerate(idx_list)
        U_vals[idx[1],idx[2],idx[3],:] .= real( field_value(y1,y2,y3,r_vals,f_plus,f_minus, idx, r_idx, Q1, Q2) )
        print("\rdone: $(round(step/length(idx_list)*100,digits=4)) %")
    end

    return U_vals
end

###

function make_field(grid_size, r_vals, Q_vals, model::String, out::String, output_format::String)

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
        return
    end

    #----- params

    IdM = [1. 0im;
           0im 1.]
    
    point_list = []
    for i in 1:length(r_vals)
        for j in 1:length(Q_vals)
            push!(point_list, [i,j])
        end
    end

    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field")
    println()

    #----- main loop

    for (idx,point) in enumerate(point_list)
        
        r_idx = point[1]
        Q_idx = point[2]

        field_grid(r_vals, rtc, r_idx, IdM, Q_vals[Q_idx,:,:], model)
        println()
        println("\rdone: r=$(r_idx), Q=$(Q_idx)")
        println()

    #----- data saving

        if output_format == "jld2"
            path = out*"/U_sym_r=$(r_idx)_Q=$(Q_idx).jld2"
            @save path U_vals

        elseif output_format == "npy"
            npzwrite(out*"/U_sym_r=$(r_idx)_Q=$(Q_idx).npy", U_vals)
        end
    end

    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")

end
