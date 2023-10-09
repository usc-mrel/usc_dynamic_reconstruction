% Prakash Kumar
% Ecrin Yagiz
%% 2D gridding recon using a "Fatrix" encoding operator.
clear; close all; clc

%% Setup paths
addpath('encoding/');

%% Select which dataset to use [See select_dataset.m]
area = 'speech';   which_file = 0;
select_dataset;

%% Recon Related Parameters here
Narms_per_frame = 13;               % [integer], input to prep function, 13 for speech, 8 for cardiac
TRtoTrim = 150;                    % [integer], input to prep function

useGPU = 1;

%% Load Data and prep
load_and_prep_data;

%% Encoding operators:

% construct encoding operator F.
F = Fnufft_2D(kx, ky, Ncoil, matrix_size, useGPU, DCF(:,1), 1, [4,4]);

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


