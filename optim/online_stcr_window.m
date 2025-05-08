%% Use online STCR to reconstruct every frame
% Written by Duc Le
% updated 04/22/2025

%---------------------------------------------------------------
function [x, out] = online_stcr_window(kspace,x_frame1, E_load, EhE_load, Narms_recon, Narms_window, lambdaTFD, lambdatTV, mu, step_size_x, Nmaxiter, cost_flag, accelerate_flag)
out.Cost = [];

NF = length(E_load);
[Nsample,~,Nt,Ncoil] = size(kspace);
T_tv = TV_2D(size(x_frame1));
% x = zeros([size(x_frame1),floor(Nt/Narms_recon)]); 

x = gpuArray(single(zeros([size(x_frame1),floor((Nt-Narms_window+Narms_recon)/Narms_recon)]))); % Reconstructed result


for i = 1:1:size(x,3)
    
    fprintf(['frame ',num2str(i), '\n'])
    kspace_fr = reshape(kspace(:,:,Narms_recon*(i-1)+(1:Narms_window),:),[Nsample,Narms_window,1,Ncoil]);
    
    % Load gridding operator
    
    E = E_load{1+mod(i-1,NF)};  
    EhE = EhE_load{1+mod(i-1,NF)};
    
    if i == 1  % Initialize first frame with gridding result (with aliasing)
        x(:,:,1) = x_frame1;
    else
    x0 = x(:,:,i-1);
    
    % ---- Solve constrained optimization problem to find temporal FD dx
    [dx, cost, cost_dual] = online_stcr_1frame_dx(x0, x0, E, EhE, T_tv, kspace_fr, lambdaTFD, lambdatTV, mu, step_size_x, Nmaxiter, cost_flag, accelerate_flag);
    out.Cost = [out.Cost; cell2mat(cost)]; 
    out.Cost_dual = cost_dual;
    if lambdatTV==0 % Online TCR
        x(:,:,i) =remove_kspace_corners(dx+x0);
%         x(:,:,i) = (dx+x0);
    else
        x(:,:,i) = (dx+x0);
    end
end
end