% 2D STCR recon using a "Fatrix" encoding operator.
% template by Prakash Kumar
% written  by Ecrin Yagiz

%   spatially and temporally constrained reconstruction (STCR):
%
%   || Em - d ||_2^2 + lambda_t || TV_t m ||_1 + lambda_s || TV_s m ||_1
%                  
%
%   "E"         sampling matrix includes sensitivity maps, Fourier 
%               transform, and undersampling mask
%   "m"         image to be reconstructed
%   "d"         measured k-space data
%   ||.||_2^2   l2 norm
%   ||.||_1     l1 norm
%   "lambda_t"  temporal constraint weight
%   "lambda_s"  sparial constraint weight
%
%   TV_t        temporal total variation (TV) operator (finite difference)
%               sqrt( abs(m_t+1 - m_t)^2 + epsilon )
%   "epsilon"   small term to aviod singularity
%   TV_s        spatial TV operator
%               sqrt( abs(m_x+1 - m_x)^2 + abs(m_y+1 - m_y) + epsilon )
%   TV_cp       TV along the cardiac dimension. image "m" is sorted.
%               sqrt(abs(m_c+1 - m_c)^2 + epsilon)
%--------------------------------------------------------------------------
%   Reference:
%       [1]     Acquisition and reconstruction of undersampled radial data 
%               for myocardial perfusion MRI. JMRI, 2009, 29(2):466-473.
%--------------------------------------------------------------------------
clear; close all; clc

%% Setup paths
restoredefaultpath
setup
addpath('encoding/');
addpath('utility/');
addpath('optim');

%% Select which dataset to use [See select_dataset.m]
area = 'cardiac';   which_file = 0;
select_dataset;

%% Recon Related Parameters here
Narms_per_frame = 8;               % [integer], input to prep function, 13 for speech, 8 for cardiac
TRtoTrim = 160;                     % [integer], input to prep function

useGPU = 1;
oversampling = 2;

%%% Solver Parameters
weight_tTV = 0.1;                  % will be scaled by the SI
weight_sTV = weight_tTV/10;

Nmaxiter    = 150;                    % Max number of iterations
Nlineiter   = 20;                     % Max number of it for Line Search
betahow     = 'GD';                   % NCG Update Methods
linesearch_how  = 'mm';        % Line Search Method
%% Load Data and prep
load_and_prep_data;

%% Encoding operators 

% 1) F -> NUFFT 
F = Fnufft_2D(kx, ky, Ncoil, matrix_size, useGPU, DCF(:,1), oversampling, [4,4]);

% --------- adjoint test on the operator F (optional). --------------------
test_fatrix_adjoint(F);

% Get coil uncombined image.
image = F' * kspace;

% 2) C -> Coil Sensitivity Operator
sens = get_sens_map(image, '2D');
C = C_2D(size(image), sens, useGPU);

% --------- adjoint test on the operator C (optional). --------------------
test_fatrix_adjoint(C);

%% First Estimate to the solver, gridding + coil combination
first_estimate = C' * image;
scale = max(abs(first_estimate(:)));

%% Regularization Operators

% operators tfd and tv.
T_tfd = TFD(size(first_estimate));
T_tv = TV_2D(size(first_estimate));

% --------- adjoint test on the operator TV (optional). -------------------
test_fatrix_adjoint(T_tfd);
test_fatrix_adjoint(T_tv);

%% Define the L1 Approximation

% Define potiential function as fair-l1.
l1_func = potential_fun('fair-l1', 0.2);    % with delta  = 0.2

%% Solver -> NCG 
% need to define B, gradF, curvf, x0, niter, ninner, P, betahow, fun

% -------------------------------------------------------------------------
% prep for the NCG routine
% -------------------------------------------------------------------------

% Scale Regularization parameters
lambdaTFD = weight_tTV * scale;
lambdatTV = weight_sTV * scale;

% ----- Data Consistency Term Related -------------------------------------
gradDC = @(x) x - kspace;
curvDC = @(x) 1;

% ------------------ TTV Term Related -------------------------------------
gradTFD = @(x) lambdaTFD * l1_func.dpot(x);
curvTFD = @(x) lambdaTFD * l1_func.wpot(x);

% ------------------ STV Term Related -------------------------------------
gradtTV = @(x) lambdatTV * l1_func.dpot(x);
curvtTV = @(x) lambdatTV * l1_func.wpot(x);

% ------------------ Intermediate Step Cost -------------------------------
costf = @(x,y) each_iter_fun(F, C, T_tfd, T_tv, lambdaTFD, lambdatTV, ...
                             l1_func, kspace, x, y);

% ------------- necessary for NCG routine ---------------------------------
B = {F*C, T_tfd, T_tv};
gradF = {gradDC, gradTFD, gradtTV};
curvF = {curvDC, curvTFD, curvtTV};

%% Actual Solver Here
tic
%[x, out] = ncg_inv_mm(B, gradF, curvF, first_estimate, 200, 20, eye,'dai-yuan', costf);
[x, out] = ncg(B, gradF, curvF, first_estimate, Nmaxiter, Nlineiter, eye, betahow, linesearch_how, costf);
toc

%% Display the Result
img_recon = gather(x);
img_recon = imrotate(fliplr(img_recon), 90);
img_recon = crop_half_FOV(img_recon);

as(img_recon);

out = cell2mat(out);
Cost = structArrayToStructWithArrays(out);
plotCost(Cost);

%% Example of how to look at the line plot
% disp('Put a Line in the Figure1 to see a line profile')
% [profiles, lines] = draw_profile_(abs(img_recon));

%% save video.
save_video("test.avi", img_recon, 0, 1000 / (kspace_info.user_TR * Narms_per_frame / 1000), false, 1/2);

%% Helper Functions
% COST
function [struct] = each_iter_fun(F, C, T_tfd, T_tv, lambdaTFD, lambdatTV, l1_func, kspace, x, y)
    % added normalization
    N = numel(x);
    struct.fidelityNorm = (0.5 * (norm(vec(F * C * x - kspace))^2)) / N;
    struct.spatialNorm = sum(vec(lambdatTV * l1_func.potk(T_tv * x))) / N;
    struct.temporalNorm = sum(vec(lambdaTFD * l1_func.potk(T_tfd * x))) / N;
    struct.totalCost = struct.fidelityNorm + struct.spatialNorm + struct.temporalNorm;

end
