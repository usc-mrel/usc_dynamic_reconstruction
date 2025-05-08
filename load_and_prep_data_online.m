
%% Load data
% file_name = fullfile(path, name);

%% Start Data prep.
Narms_per_frame = 1;

res = [kspace_info.user_ResolutionX, kspace_info.user_ResolutionY];         % [mm]
matrix_size = ceil([kspace_info.user_FieldOfViewX, kspace_info.user_FieldOfViewX] ./ res);  
viewOrder = kspace_info.viewOrder;

% kspace and trajectory
kspace = permute(kspace, [1, 2, 4, 3]);
kx = kspace_info.kx_GIRF;
ky = kspace_info.ky_GIRF;
% kx = kspace_info.kx;
% ky = kspace_info.ky;

% trim TR for steady state.
if ~exist('TRtoTrim_end', 'var')
    TRtoTrim_end = 0;
end

%% Load kspace...
kspace = kspace(:, TRtoTrim+1:end-TRtoTrim_end, :, :);
viewOrder = viewOrder(TRtoTrim+1:end-TRtoTrim_end);

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

% reshape kspace
kspace = reshape(kspace, [Nsample, Narms_per_frame, Nframes, Ncoil]);

Nsample_k = size(kx, 1);

% reshape trajectory
kx = reshape(kx, [Nsample_k, Narms_per_frame, Nframes]);
ky = reshape(ky, [Nsample_k, Narms_per_frame, Nframes]);

% DCF 
DCF = kspace_info.DCF;

% pre-weight kspace by sqrt(DCF)
kspace = kspace .* sqrt(DCF(:,1));

%% First Narms_full frames
kx_full = reshape(kx(:,:,1:Narms_full),[Nsample, Narms_full]);
ky_full = reshape(ky(:,:,1:Narms_full),[Nsample, Narms_full]);
kspace_full = reshape(kspace(:,:,1:Narms_full,:),[Nsample, Narms_full,1,Ncoil]);
%% Initial frame
kx_initial = reshape(kx(:,:,1:Narms_initial),[Nsample, Narms_initial]);
ky_initial = reshape(ky(:,:,1:Narms_initial),[Nsample, Narms_initial]);
kspace_initial = reshape(kspace(:,:,1:Narms_initial,:),[Nsample, Narms_initial,1, Ncoil]);
%% Frames to reconstruct
kx = kx(:,:,1+max(Narms_full,Narms_initial):end);
ky = ky(:,:,1+max(Narms_full,Narms_initial):end);
kspace = kspace(:,:,1+max(Narms_full,Narms_initial):end,:);