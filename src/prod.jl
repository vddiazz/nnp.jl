#----- pkg

using JLD2
using NPZ
using LinearAlgebra
using SpecialFunctions

using ProgressMeter
using LoopVectorization

#----- field

function field_value(y1,y2,y3,r_val,f_plus,f_minus, coord::Vector{Int64}, Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64})

    #----- params

    sigma = [[0im 1.; 1. 0im],
             [0. -1im; 1im 0.],
             [1. 0im; 0im -1.]]

    y = [y1[coord[1]], y2[coord[2]], y3[coord[3]]]
    xm = [0., 0., r_val/2.]
    pos_p = y.+xm; pos_m = y.-xm

    ar_p = norm(pos_p, 2)
    ar_m = norm(pos_m, 2)

    #----- f(r) values

    f_p = f_plus[coord[1],coord[2],coord[3],1]
    f_m = f_minus[coord[1],coord[2],coord[3],1]

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

function field_grid(rtc, f_plus,f_minus, r_val::Float64, Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare rtc and params

    y1 = rtc[1]; y2 = rtc[2]; y3 = rtc[3]

    #----- evaluate field values
    
    U_vals = zeros(Float64, length(y1),length(y2),length(y3),4)

    step = 0
    for k in 1:length(y3)
        for j in 1:length(y2)
            for i in length(y1)
                U_vals[i,j,k,:] .= real( field_value(y1,y2,y3,r_val,f_plus,f_minus, [i,j,k], Q1, Q2) )
                step =+ 1
                print("\rdone: $(round(step/(length(y1)*length(y2)*length(y3))*100,digits=4)) %")
            end
        end
    end

    return U_vals
end

###

function make_field(grid_size, r_vals, Q_vals, model::String, out::String, output_format::String)

    if (output_format != "jld2") && (output_format != "npy")
        println("invalid output data type")
    end

    #----- params

    y1 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y1.npy")
    y2 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y2.npy")
    y3 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y3.npy")

    rtc = [y1,y2,y3]
    
    f_plus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_plus.npy")
    f_minus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_minus.npy")

    IdM = [1. 0im;
           0im 1.]

    println()
    println("#--------------------------------------------------#")
    println()
    println("2-skyrmion field")
    println()

    #----- main loop

    for Q_idx in 1:length(Q_vals)
        for r_idx in 1:length(r_vals)
            
            field_grid(r_vals[r_idx], rtc, r_idx, IdM, Q_vals[Q_idx,:,:], model)
            
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
    end

    println()
    println("data saved at "*out )
    println()
    println("#--------------------------------------------------#")

end

###

function field_grid_single(grid_size, r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare rtc and params

    y1 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y1.npy")
    y2 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y2.npy")
    y3 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y3.npy")

    rtc = [y1,y2,y3]

    f_minus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_minus_r=61.npy")
    f_plus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_plus_r=61.npy")

    #----- evaluate field values
    
    U_vals = zeros(Float64, length(y1),length(y2),length(y3),4)

    step = 0
    for k in 1:length(y3)
        for j in 1:length(y2)
            for i in 1:length(y1)
                U_vals[i,j,k,:] .= real( field_value(y1,y2,y3,r_val,f_plus,f_minus, [i,j,k], Q1, Q2) )
                step = step+1
                print("\rdone: $(round(step/(length(y1)*length(y2)*length(y3))*100,digits=4)) %")
            end
        end
    end

    return U_vals
end

###

function FAST_field_grid_single(grid_size, r_val::Float64,Q1::Matrix{ComplexF64}, Q2::Matrix{ComplexF64}, model::String)

    #----- prepare rtc and params

    y1 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y1.npy")
    y2 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y2.npy")
    y3 = npzread("/home/velni/phd/w/nnp/data/sample/$(grid_size)/y3.npy")

    rtc = [y1,y2,y3]

    f_minus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_minus_r=61.npy")
    f_plus = npzread("/home/velni/phd/w/nnp/data/interp/$(model)/$(grid_size)/f_$(model)_plus_r=61.npy")
    
    sigma1 = [0im 1.; 1. 0im]
    sigma2 = [0. -1im; 1im 0.]
    sigma3 = [1. 0im; 0im -1.]
    sigma = [sigma1, sigma2, sigma3]

    #----- prelocate arrays

    y = zeros(Float64, 3)
    xm = zeros(Float64, 3)
    pos_p = zeros(Float64, 3)
    pos_m = zeros(Float64, 3)
    
    dirs1 = [(Q1*s)*inv(Q1) for s in sigma]
    dirs1_param = zeros(ComplexF64, 4)
    phi1 = zeros(ComplexF64, 4)

    dirs2 = [(Q2*s)*inv(Q2) for s in sigma]
    dirs2_param = zeros(ComplexF64, 4)
    phi2 = zeros(ComplexF64, 4)

    U = zeros(ComplexF64, 4)

    #----- evaluate field values
    
    U_vals = zeros(Float64, length(y1),length(y2),length(y3),4)

    @showprogress 1 "Computing..." for k in 1:length(y3)
        @tturbo for j in 1:length(y2), i in 1:length(y1) # broadcast sigma1 .* Z1 not working
               
            #----- params        

            y[1] = y1[i]; y[2] = y2[j]; y[3] = y3[k]
            xm[3] = r_val/2.
            pos_p[1] = y[1]+xm[1]; pos_p[2] = y[2]+xm[2]; pos_p[3] = y[3]+xm[3]
            pos_m[1] = y[1]-xm[1]; pos_m[2] = y[2]-xm[2]; pos_m[3] = y[3]-xm[3]

            ar_p = norm(pos_p, 2)
            ar_m = norm(pos_m, 2)

            #----- f(r) values

            f_p = f_plus[i,j,k,1]
            f_m = f_minus[i,j,k,1]

            cos_f_p = 1 - 0.5*f_p^2 + (1/24.)*f_p^4 - (1/720.)*f_p^6
            cos_f_m = 1 - 0.5*f_m^2 + (1/24.)*f_m^4 - (1/720.)*f_m^6
            sin_f_p = f_p - (1/6.)*f_p^3 + (1/120.)*f_p^5 - (1/5040.)*f_p^7
            sin_f_m = f_m - (1/6.)*f_m^3 + (1/120.)*f_m^5 - (1/5040.)*f_m^7

            #----- U1

            Z1 = pos_m[1]*dirs1[1] + pos_m[2]*dirs1[2] + pos_m[3]*dirs1[3]
            dirs1_param[1] = 0.5*(Z1[1]+Z1[2]); dirs1_param[2] = 0.5*((sigma1.*Z1)[1]+(sigma1.*Z1)[2]); dirs1_param[3] = 0.5*( (sigma2.*Z1)[1]+(sigma2.*Z1)[2] ); dirs1_param[4] = 0.5*((sigma3.*Z1)[1] + (sigma3.*Z1)[2])
            phi1[1] = cos_f_m+dirs1_param[1]; phi1[2] = sin_f_m*(1/ar_m)*dirs1_param[2]; phi1[3] = sin_f_m*(1/ar_m)*dirs1_param[3]; phi1[4] = sin_f_m*(1/ar_m)*dirs1_param[4]

            #----- U2

            Z2 = pos_p[1]*dirs2[1] + pos_p[2]*dirs2[2] + pos_p[3]*dirs2[3]
            dirs2_param[1] = 0.5*(Z2[1]+Z2[2]); dirs2_param[2] = 0.5*((sigma1.*Z2)[1]+(sigma1.*Z2)[2]); dirs2_param[3] = 0.5*((sigma2.*Z2)[1]+(sigma2.*Z2)[2]); dirs2_param[4] = 0.5*((sigma3.*Z2)[1] + (sigma3.*Z2)[2])
            phi2[1] = cos_f_p+dirs2_param[1]; phi2[2] = sin_f_p*(1/ar_p)*dirs2_param[2]; phi2[3] = sin_f_p*(1/ar_p)*dirs2_param[3]; phi2[4] = sin_f_p*(1/ar_p)*dirs2_param[4]

            #----- symmetrized product field

            U[1] = phi1[1]*phi2[1] - phi1[2]*phi2[2] - phi1[3]*phi2[3] - phi1[4]*phi2[4]
            U[2] = phi1[1]*phi2[2] + phi1[2]*phi2[1]
            U[3] = phi1[1]*phi2[3] + phi1[3]*phi2[1]
            U[4] = phi1[1]*phi2[4] + phi1[4]*phi2[1]

            #----- normalization

            C0 = (U[1])^2; Ck = U[2]^2 + U[3]^2 + U[4]^2
            N = sqrt(C0 + Ck)

            #----- return

            U_vals[i,j,k,1] = real( (1/N)*U[1] ); U_vals[i,j,k,2] = real( (1/N)*U[2] ); U_vals[i,j,k,3] = real( (1/N)*U[3] ); U_vals[i,j,k,4] = real( (1/N)*U[4] )
        
        end
    end

    return U_vals
end
