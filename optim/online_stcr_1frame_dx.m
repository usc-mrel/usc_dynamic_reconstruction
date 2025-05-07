%% Use online STCR to reconstruct 1 frame
% Written by Duc Le
% updated 04/22/2025

function [dx, cost, cost_dual] = online_stcr_1frame_dx(x0, xinit, E, EhE, T_tv, kspace, lambdaTFD, lambdatTV, mu, step_size_x,Nmaxiter, cost_flag, accelerate_flag)
x_grid = E'*kspace;
x = xinit;
y = x;
dx = xinit-x0; % x-x0
z = T_tv*(x0+dx); % sTV(x)
t = 1;    
ydx = dx;
yz = z;    
eta = gpuArray(single(zeros(size(z))));    
iter = 0;
N = numel(dx);

cost = cell(1,Nmaxiter+1);
cost_dual = cell(1,Nmaxiter+1);

if cost_flag
[cost{1}, cost_dual{1}] = cost_fun(E, dx, x0, z, eta, kspace, T_tv, lambdaTFD, lambdatTV, mu);
fprintf(sprintf('%10.0f %10.4f %10.4f %10.4f %10.4f \n',iter,cost{1}.fidelityNorm, cost{1}.temporalNorm, cost{1}.spatialNorm, cost{1}.totalCost));
end



% --- Start optimization ----------
for iter = 1:1:Nmaxiter

%         fprintf(['iter ', num2str(iter),'\n']);

% -------------------Update dx ------------------------------
tic
if accelerate_flag
    t_new = (1+sqrt(1+4*t^2))/2;
    shrink_arg_dx = ydx - step_size_x*(EhE*(dx+x0)-x_grid + T_tv'*(T_tv*(dx+x0)-z+eta)*mu); % Toeplitz update
    dx_new = shrinkage(shrink_arg_dx, lambdaTFD*step_size_x);           
    ydx = dx_new+(t-1)/(t_new+1)*(dx_new-dx);    

    t = t_new;
else 
    shrink_arg_dx = dx - step_size_x*(EhE*(dx+x0)-x_grid + T_tv'*(T_tv*(dx+x0)-z+eta)*mu); % Toeplitz update
    dx_new = shrinkage(shrink_arg_dx, lambdaTFD*step_size_x);   
end
dx = dx_new;
if iter < Nmaxiter
% ---------Update z=~sTV(x) --------------------------------
stv = T_tv*(dx+x0);
z = shrinkage(stv+eta, lambdatTV/mu); 
% ------------Update dual eta --------------------------------
eta = eta - z + stv;

end  
toc
% Calculate and print cost        
if cost_flag
    [cost{iter+1}, cost_dual{iter+1}] = cost_fun(E, dx, x0, z, eta, kspace, T_tv, lambdaTFD, lambdatTV, mu);
    fprintf(sprintf('%10.0f %10.4f %10.4f %10.4f %10.4f \n',iter,cost{iter+1}.fidelityNorm, cost{iter+1}.temporalNorm, cost{iter+1}.spatialNorm, cost{iter+1}.totalCost));
end

end
% dx = x-x0;
end

function [cost, cost_dual] = cost_fun(E, dx, x0, z, eta, kspace, T_tv, lambdaTFD, lambdatTV, mu)
N = numel(dx);
cost.fidelityNorm = 1/2*norm(vec(E*(dx+x0)-kspace))^2/N;
cost.temporalNorm = lambdaTFD*sum(abs(vec(dx)))/N;
cost.spatialNorm = lambdatTV*sum(abs(vec(T_tv*(dx+x0))))/N;
cost.totalCost = cost.fidelityNorm + cost.temporalNorm + cost.spatialNorm ;

cost_dual.fidelityNorm = 1/2*norm(vec(E*(dx+x0)-kspace))^2/N;
cost_dual.temporalNorm = lambdaTFD*sum(abs(vec(dx)))/N;
cost_dual.spatialNorm = lambdatTV*sum(abs(vec(z)))/N;
cost_dual.dualNorm = mu/2*(norm(vec(T_tv*(dx+x0)-z+eta))^2+norm(vec(eta))^2)/N;
cost_dual.totalCost = cost_dual.fidelityNorm + cost_dual.temporalNorm + cost_dual.spatialNorm +cost_dual.dualNorm;
end