% 2D gridding recon using a "Fatrix" encoding operator.
clear; close all; clc

%% load data
path = "/server/home/pkumar/mri_data/disc/speech/vol0634_20230601/raw_hawk/";
name = "usc_disc_20230601_172615_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr.mat";
file_name = path + name;
load(file_name);

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
addpath('encoding/');

% construct encoding operator F.
F = Fnufft_2D(kx, ky, Ncoil, matrix_size, DCF(:,1), 1, [4,4]);

% adjoint test on the operator F (optional).
test_fatrix_adjoint(F);

%pre-weight kspace by sqrt(DCF) and then encode into image.
kspace = kspace .* sqrt(DCF(:,1));
image = F' * kspace;

% estimate sensitivity maps
sens = get_sens_map(image, '2D');

% C = 2D coil operator.
C = C_2D(size(image), sens);
image_coil_combined = C' * image;




