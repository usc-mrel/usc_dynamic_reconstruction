%% Reconstruction for fully-sampled "snapshot" multiplanar spiral bSSFP 
% Written by Duc Le
% updated 08/17/2026

% Load raw multiplanar kspace data (from RTHawk) and perform gridding
% reconstruction

%% Load path
currentFile = mfilename('fullpath');
[currentPath, name, ext] = fileparts(currentFile);
addpath(currentPath);
addpath([currentPath,'/encoding/']);
addpath([currentPath,'/utility/']);
addpath([currentPath,'/optim/']);

%% Prepare raw data
sprintf("-------------------------------------------------")
useGPU = false;
oversampling = 2;

Ntr_per_slice = 25;          % Number of TR/frame/slice
Nprep = 0;                  % Number of TR used for preparation, ignored
Nreadout_per_slice = 25;     % Number of TR actually used for readout

FramestoTrim = 0;         % Trim frames in the beginning
FramestoTrim_end = 0;     % Trim frames at the end

fprintf("Prepare raw data\n")
load_and_prep_data_multiplanar;

% Turn coils on or off
coils = cell(Nslice,1);
coils{1} = [1,2,3,4,5,6,7,9,10,12];
coils{2} = [1,2,3,4,6,7,8,9,10,11,12];
coils{3} = [1,2,3,4,6,7,8,9,10,11,12];

%% Recon
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
    % First Estimate to the solver, gridding + coil combination

    fprintf(" Load gridding operators \n")
    
    F = Fnufft_2D(kx(:,:,:), ky(:,:,:), Ncoil, matrix_size, useGPU, DCF(:,1).^1, oversampling, [4,4]);

    fprintf(" Gridding + Coil combination \n")
    image = F' * kspace(:,:,:,:);
    
    img{slice} = image;
    
    sens = get_sens_map(image, '2D');
    C = C_2D(size(image), sens, useGPU);
    
    x{slice} = C' * image;
    
end

%% Stitch images from 2 slices into a video
xdisp = cell(Nslice,1);
for i = 1:Nslice
    xdisp{i} = crop_half_FOV(rot90(fliplr((x{i}))));
end
D = size(xdisp{1},1);
video = zeros(size(xdisp{1}).*[1,Nslice,Nslice]+[0,0,Nslice-1]);
for slice = 1:1:Nslice
    for fr = 1:size(x{1},3)
        video(:,(slice-1)*D+(1:D),slice-1+(fr-1)*Nslice+(1:Nslice)) = repmat(xdisp{slice}(:,:,fr),[1,1,Nslice]);
    end
end
%% Display final result
implay(abs(video)/prctile(abs(video(:)),95), 1e6/(Nreadout_per_slice+2)/kspace_info.user_TR)
