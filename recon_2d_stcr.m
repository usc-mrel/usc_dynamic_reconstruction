% 2D STCR recon using a "Fatrix" encoding operator.
% written by Prakash Kumar

clear; close all; clc

addpath('encoding/');
addpath('utility/');

%% load data
path = "/server/home/pkumar/mri_data/disc/speech/vol0634_20230601/raw_hawk/";
name = "usc_disc_20230601_172615_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr.mat";
file_name = path + name;
load(file_name);

% important flag: use GPU or CPU.
useGPU = 1;

%% Data prep.
Narms_per_frame = 13; % parameter! can be changed.
res = [kspace_info.user_ResolutionX, kspace_info.user_ResolutionY];
matrix_size = round([kspace_info.user_FieldOfViewX, kspace_info.user_FieldOfViewY] ./ res);
viewOrder = kspace_info.viewOrder;

kspace = permute(kspace, [1, 2, 4, 3]);
kx = kspace_info.kx_GIRF;
ky = kspace_info.ky_GIRF;

% trim TR for steady state.
TRtoTrim = 150; % parameter! can be changed.
kspace = kspace(:, TRtoTrim+1:end, :, :);
viewOrder = viewOrder(TRtoTrim+1:end);

GA_steps = size(kx, 2);
Narms_total = size(kspace, 2);
Nframes = floor(Narms_total / Narms_per_frame);
Narms_total = Nframes * Narms_per_frame;
Ncoil = size(kspace, 4);
Nsample = size(kspace, 1);

kx = repmat(kx, [1, ceil(Narms_total / GA_steps)]);
ky = repmat(ky, [1, ceil(Narms_total / GA_steps)]);

kspace(:, Narms_total + 1 : end, :, :) = [];
viewOrder(Narms_total + 1 : end) = [];

kx = kx(:, viewOrder);
ky = ky(:, viewOrder);

kspace = reshape(kspace, [Nsample, Narms_per_frame, Nframes, Ncoil]);

Nsample_k = size(kx, 1);
kx = reshape(kx, [Nsample_k, Narms_per_frame, Nframes]);
ky = reshape(ky, [Nsample_k, Narms_per_frame, Nframes]);
DCF = kspace_info.DCF;

%% Encoding operators:
% construct encoding operator F.
F = Fnufft_2D(kx, ky, Ncoil, matrix_size, useGPU, ones(size(DCF(:,1))), 2, [4,4]);

% adjoint test on the operator F (optional).
test_fatrix_adjoint(F);

%pre-weight kspace by sqrt(DCF) and then encode into image.
kspace = kspace .* sqrt(DCF(:,1));
image = F' * kspace;

% estimate sensitivity maps
sens = get_sens_map(image, '2D');

% C = 2D coil operator.
C = C_2D(size(image), sens, useGPU);
image_coil_combined = C' * image;
test_fatrix_adjoint(C);

scale = max(abs(image_coil_combined(:)));


%% to run NCG, need to define B, gradF, curvf, x0, niter, ninner, P, betahow, fun
addpath('optim');

% Define potiential function as fair-l1.
l1_func = potential_fun('fair-l1', 0.2);

% operators tfd and tv.
T_tfd = TFD(size(image_coil_combined));
T_tv = TV_2D(size(image_coil_combined));

% test adjoint (optional)
test_fatrix_adjoint(T_tfd);
test_fatrix_adjoint(T_tv);

% define regularization paramters
lambdaTFD = 0.15 * scale;
lambdatTV = 0;

gradDC = @(x) x - kspace;
curvDC = @(x) 1;

gradTFD = @(x) lambdaTFD * l1_func.dpot(x);
gradtTV = @(x) lambdatTV * l1_func.dpot(x);

curvTFD = @(x) lambdaTFD * l1_func.wpot(x);
curvtTV = @(x) lambdatTV * l1_func.wpot(x);

% costf = @(x,y) (0.5 * (norm(vec(F * C * x - kspace))^2) + sum(vec(lambdaTFD * l1_func.potk(T_tfd * x))) + sum(vec(lambdatTV * l1_func.potk(T_tv * x))));
costf = @(x,y) each_iter_fun(F,C,T_tfd,T_tv,lambdaTFD,lambdatTV,l1_func,kspace,x,y);

B = {F*C, T_tfd, T_tv};
gradF = {gradDC, gradTFD, gradtTV};
curvF = {curvDC, curvTFD, curvtTV};

if useGPU
   image_coil_combined = gpuArray(image_coil_combined);
end

tic
[x, out] = ncg_inv_mm(B, gradF, curvF, image_coil_combined, 200, 20, eye,'dai-yuan', costf);
toc

img_recon = gather(x);
img_recon = imrotate(fliplr(img_recon), 90);
img_recon = crop_half_FOV(img_recon);

as(img_recon);

out = cell2mat(out);
Cost = structArrayToStructWithArrays(out);
plotCost(Cost);

% save video.
save_video("test.avi", img_recon, 0, 1000 / (kspace_info.user_TR * Narms_per_frame / 1000), false, 1/2);

function [struct] = each_iter_fun(F, C, T_tfd, T_tv, lambdaTFD, lambdatTV, l1_func, kspace, x, y)
    struct.fidelityNorm = (0.5 * (norm(vec(F * C * x - kspace))^2));
    struct.spatialNorm = sum(vec(lambdatTV * l1_func.potk(T_tv * x)));
    struct.temporalNorm = sum(vec(lambdaTFD * l1_func.potk(T_tfd * x)));
    struct.totalCost = struct.fidelityNorm + struct.spatialNorm + struct.temporalNorm;
    fprintf(" ------ iteration %i, cost: %d -------- \n", y, struct.totalCost);
end
