%% Main online STCR
% Written by Duc Le
% updated 04/22/2025

%   Online spatiotemporally constrained reconstruction (online STCR):
%
%   ||E_n x_n-d_n||_2^2 + lambda_t||x_n-x_n-1||_1 + lambda_s||TV_s x_n||_1
%                  
%
%   "E_n"       sampling matrix includes sensitivity maps, Fourier 
%               transform, and undersampling mask for nth frame
%   "x_n"       nth frame to be reconstructed
%   "x_n-1"     Reconstructed previous frame
%   "d_n"       measured k-space  data of nth frame
%   ||.||_2^2   l2 norm
%   ||.||_1     l1 norm
%   "lambda_t"  temporal constraint weight
%   "lambda_s"  spatial constraint weight
%
%   TV_s        spatial TV operator
%               sqrt( abs(m_x+1 - m_x)^2 + abs(m_y+1 - m_y) + epsilon )
%-------------------------------------------------------------------------
% clear
restoredefaultpath
currentFile = mfilename('fullpath');
[currentPath, name, ext] = fileparts(currentFile);
addpath(currentPath);
addpath([currentPath,'/encoding/']);
addpath([currentPath,'/utility/']);
addpath([currentPath,'/optim/']);
addpath([currentPath,'/mirt/systems/']);
addpath([currentPath,'/mirt/utilities/']);

% addpath('../onlineSTCR_repo/arrShow/');

%% Load data
% load raw kspace acqusition data
% area = 'cardiac';   which_file = 0;
% select_dataset;
% load('volunteer1.mat'); 
%% Trim first and last few TRs where steady state is not reached
TRtoTrim = 30;
TRtoTrim_end = 0; 
%% Recon Related Parameters
Narms_full = 34; % Number of spirals for full sampling 
Narms_recon = 6; % Desired temporal resolution (distance between frames)
Narms_window = 10; % How many TRs to bin in 1 frame
Narms_initial = 55; % How many TRs to bin for initial frame
%% Optimization Parameters
lambdaTFD = 1; % Temporal regularization
lambdatTV = 0.2; % Spatial regularization
mu = 0.1; % Penalty of Lagrange multiplier
%% Solver parameters
step_size_x =2; % step size of shrinkage operator 
Nmaxiter = 6; % number of iterations per frame
print_cost = true; % print cost of optimization problem each frame (takes more time)
accelerate_flag = true; % apply Nesterov acceleration
useGPU = true;
toeplitz_flag = false; % Use Toeplitz to make E'*E faster
oversampling = 2; % Oversampling FOV when doing NUFFT
%% Load Data and prep
load_and_prep_data_online;
Nsamples = size(kspace,1); % Number of samples per spiral arm (1TR)
Ncoil = size(kspace,4); % Number of coils
Nt = round(size(kspace,3)); % Number of time frames
%% Coil combination operator
% Fully reconstruct 1 frame from first Narms_full TRs
F_full = Fnufft_2D(kx_full, ky_full, Ncoil, matrix_size, useGPU, DCF(:,1), oversampling, [4,4]);
image_full = F_full'*kspace_full; % Gridded image of each coil
sense_full = get_sens_map(image_full, '2D'); % Find sensitivity map
C = C_2D(size(image_full), sense_full, useGPU); % Use coil combination calculated from fully sampled frame for reconstruction 
%% Initial frame x1
F1 = Fnufft_2D(kx_initial, ky_initial, Ncoil, matrix_size, useGPU, DCF(:,1), oversampling, [4,4]);
% Gridded result 
x1 = C'*F1'*kspace_initial;
% Rescale gridded result by minimizing fidelity norm with kspace measurements
x1 = x1*abs(vec(kspace_initial)'*vec(F1*C*x1)/norm(vec(F1*C*x1))^2);
% x1 = gpuArray(zeros(size(x1))); % Zero initialization;
%% Precompute gridding operator
preload_nufft_window; % Preload NUFFT operators of every frame
%% -------- Online STCR -------------
[x, out] = online_stcr_window(kspace,x1, E_load, EhE_load, Narms_recon, Narms_window, lambdaTFD, lambdatTV, mu, step_size_x, Nmaxiter, print_cost, accelerate_flag);
%% Display the Result
% Shows video of result
result = abs(gather(x));
implay(crop_half_FOV(imrotate(fliplr(result), 90))/quantile(vec(result),0.99), 1e6/(kspace_info.user_TR*Narms_recon))
% Display cost evolution of reconstructing last frame
if print_cost
    figure
    plotCostLin(structArrayToStructWithArrays(out.Cost(end,:)));
end
Cost = out.Cost;