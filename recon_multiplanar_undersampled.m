%% Reconstruction for undersampled "snapshot" multiplanar spiral bSSFP 
% Written by Duc Le
% updated 08/17/2026

% Load raw multiplanar kspace data (from RTHawk) and perform
% compressed-sensing spatiotemporally constrained reconstruction (STCR) for
% each slice

%% Load path
currentFile = mfilename('fullpath');
[currentPath, name, ext] = fileparts(currentFile);
addpath(currentPath);
addpath([currentPath,'/encoding/']);
addpath([currentPath,'/utility/']);
addpath([currentPath,'/optim/']);

%% Prepare raw data 
sprintf("-------------------------------------------------")
useGPU = false;             % Set false if not having GPU toolbox
oversampling = 2;           % Oversampling in NUFFT (default = 2)

Ntr_per_slice = 5;          % Number of TR/frame/slice
Nprep = 0;                  % Number of TR used for preparation, ignored
Nreadout_per_slice = 5;     % Number of TR actually used for readout

FramestoTrim = 500;         % Trim frames in the beginning
FramestoTrim_end = 500;     % Trim frames at the end

fprintf("--- Prepare raw data ---\n")
load_and_prep_data_multiplanar;
%% Undersampled real-time recon parameters 
lambdaTFD = 3;                  % Temporal TV weight
lambdatTV = 0.0;                % Spatial TV weight
Nmaxiter    = 20;              % Max number of iterations
Nlineiter   = 10;              % Max number of it for Line Search
betahow     = 'YT';            % NCG Update Methods
linesearch_how  = 'mm';        % Line Search Method

%% Reconstruction

% Turn coils on or off
coils = cell(Nslice,1);
coils{1} = 1:Ncoil;
coils{2} = 1:Ncoil;
coils{3} = 1:Ncoil;

x = cell(Nslice,1);
clear kspace_all
img = cell(Nslice,1);
for slice = 1:1:Nslice
    % Select slice raw data and trajectory
    fprintf("--- Slice %d ---\n", slice)
    kspace = kspace_seq(:,:,:,coils{slice},slice);
    Ncoil = size(kspace,4);
    kx = kx_seq(:,:,:,slice);
    ky = ky_seq(:,:,:,slice);

    % Gridding + coil combination operators

    fprintf(" Load gridding operators \n")
    
    F = Fnufft_2D(kx(:,:,:), ky(:,:,:), Ncoil, matrix_size, useGPU, DCF(:,1).^1, oversampling, [4,4]);

    fprintf(" First estimate \n")
    image = F' * kspace(:,:,:,:);
    
    img{slice} = image;

    fprintf(" Load coil combination operators \n")
    
    sens = get_sens_map(image, '2D');
    C = C_2D(size(image), sens, useGPU);
    
    % Calculate first estimate for NCG solver (gridding, with aliasing)
    fprintf(" Calculate first estimate \n")
    first_estimate_prescale = C' * image;
    kspace_prescale = F*C*first_estimate_prescale;
    first_estimate = first_estimate_prescale*abs(vec(kspace)'*vec(kspace_prescale)/norm(vec(kspace_prescale))^2); % magnitude scaling

    Nframes = size(first_estimate,3);

    % Regularization Operators
    T_tfd = TFD(size(first_estimate));
    T_tv = TV_2D(size(first_estimate));
    
    % Define potiential function as fair-l1.
    l1_func = potential_fun('fair-l1', 0.2);    % with delta  = 0.2

    % Input operators for NCG solver
    gradDC = @(x) x - kspace;
    curvDC = @(x) 1;
    gradTFD = @(x) lambdaTFD * l1_func.dpot(x);
    curvTFD = @(x) lambdaTFD * l1_func.wpot(x);
    gradtTV = @(x) lambdatTV * l1_func.dpot(x);
    curvtTV = @(x) lambdatTV * l1_func.wpot(x);
    costf = @(x,y) each_iter_fun(F, C, T_tfd, T_tv, lambdaTFD, lambdatTV, ...
                                 l1_func, kspace, x, y);
    B = {F*C, T_tfd, T_tv};
    gradF = {gradDC, gradTFD, gradtTV};
    curvF = {curvDC, curvTFD, curvtTV};

    % ----- Actual NCG Solver Here: -----
    [x{slice}, out] = ncg(B, gradF, curvF, first_estimate, Nmaxiter, Nlineiter, eye, betahow, linesearch_how, costf);
end

%% Stitch images from 2 slices into a video
xdisp = cell(Nslice,1);
for i = 1:Nslice
    xdisp{i} = crop_half_FOV(rot90(fliplr((x{i}))));
end
D = size(xdisp{1},1);
video = zeros(size(xdisp{1}).*[1,Nslice,Nslice]+[0,0,Nslice-1]);
for slice = 1:1:Nslice
    for fr = 1:Nframes
        video(:,(slice-1)*D+(1:D),slice-1+(fr-1)*Nslice+(1:Nslice)) = repmat(xdisp{slice}(:,:,fr),[1,1,Nslice]);
    end
end
%% Display final result
implay(abs(video)/prctile(abs(video(:)),95), 1e6/(Nreadout_per_slice+2)/kspace_info.user_TR)
%% Cost functions
function [struct] = each_iter_fun(F, C, T_tfd, T_tv, lambdaTFD, lambdatTV, l1_func, kspace, x, y)
    % added normalization
    N = numel(x);
    struct.fidelityNorm = (0.5 * (norm(vec(F * C * x - kspace))^2)) / N;
    struct.temporalNorm = sum(vec(abs(lambdaTFD*(T_tfd*x))))/N;
    struct.spatialNorm = sum(vec(abs(lambdatTV*(T_tv*x))))/N;
    struct.totalCost = struct.fidelityNorm + struct.spatialNorm + struct.temporalNorm;
end
