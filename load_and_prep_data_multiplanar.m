
% Load dataset and reshape,scale accordingly
%% Load data
% file_name = fullfile(path, name);
% load(file_name);
%% Start Data prep.
Nslice = size(kspace_info.kx_GIRF,3);
% Nslice = 3;


% kspace_info.user_ResolutionX = 2.5;
% kspace_info.user_ResolutionY = 2.5;
kspace_info.user_ResolutionX = 2.5;
kspace_info.user_ResolutionY = 2.5;
res = [kspace_info.user_ResolutionX, kspace_info.user_ResolutionY];         % [mm]
matrix_size = ceil([kspace_info.user_FieldOfViewX, kspace_info.user_FieldOfViewX] ./ res);  
viewOrder_all = kspace_info.viewOrder;

% kspace and trajectory
kspace_all = permute(kspace, [1, 2, 4, 3]);
Narms_unique = size(kspace_info.kx,2);
% Ntr_per_slice = kspace_info.nTR_per_slice; 
% Ntr_per_slice = 26;
% Nprep = kspace_info.nPrep; % Number of preparation pulses (pre and post)
% Nprep = 5;
% Nreadout_per_slice = Ntr_per_slice - 2*Nprep;
% idx_zero = find(diff(kspace_info.viewOrder)==0);
% idx_first_diff = find(diff(idx_zero)~=1,1);
% Nreadout_per_slice = idx_zero(idx_first_diff+1)-idx_zero(idx_first_diff);
% clear idx_zero idx_first_diff
% Nreadout_per_slice = 20;
Nframes = floor(size(kspace_all,2)/Ntr_per_slice/Nslice);

kx_traj = kspace_info.kx_GIRF;
ky_traj = kspace_info.ky_GIRF;
% kx_traj = kspace_info.kx;
% ky_traj = kspace_info.ky;

% trim TR for steady state.
% if ~exist('kspace_info{1}_end', 'var')
%     FramestoTrim_end = 0;
% end
% kspace = kspace(:, FramestoTrim+1:end-FramestoTrim_end, :, :);
% viewOrder = viewOrder(FramestoTrim+1:end-FramestoTrim_end);
% Nprep = (Ntr_per_slice-Narms_unique)/2;


kx_seq = [];
ky_seq = [];
kspace_seq = [];

for i = 1:1:Nslice
GA_steps = size(kx_traj, 2);
Narms_total = size(kspace_all, 2);
% Nframes = floor(Narms_total / Narms_per_frame);
% Narms_total = Nframes * Narms_per_frame;
Ncoil = size(kspace_all, 4);
Nsample = size(kspace_all, 1);

viewOrder = [];
kspace = [];
idx = [];
for j = 1:1:Nframes
    idx = [idx,Ntr_per_slice*(i-1)+Ntr_per_slice*Nslice*(j-1)+Nprep+(1:Nreadout_per_slice)];
%     disp(j)
    % viewOrder = [viewOrder,viewOrder_all(Ntr_per_slice*(i-1)+Ntr_per_slice*Nslice*(j-1)+Nprep+(1:Nreadout_per_slice))];
	% kspace = cat(2,kspace,kspace_all(:,Ntr_per_slice*(i-1)+Ntr_per_slice*Nslice*(j-1)+Nprep+(1:Nreadout_per_slice),:,:));
end
viewOrder = viewOrder_all(idx);
kspace = kspace_all(:,idx,:,:);

idx_frame = Ntr_per_slice*(i-1) + [1:Ntr_per_slice]' + Ntr_per_slice*Nslice*[0:Nframes-1];
% kspace = kspace_all(:,reshape(idx_frame,[numel(idx_frame),1]),:,:);

% kx = repmat(kx, [1, ceil(Narms_total / GA_steps)]);
% ky = repmat(ky, [1, ceil(Narms_total / GA_steps)]);

kx = repmat(kx_traj(:,:,i),[1,Nframes]);
ky = repmat(ky_traj(:,:,i),[1,Nframes]);

% kx = repmat(kx_traj(:,:),[1,Nframes]);
% ky = repmat(ky_traj(:,:),[1,Nframes]);

% kspace(:, Narms_total + 1 : end, :, :) = [];
% viewOrder(Narms_total + 1 : end) = [];

kx = kx(:, viewOrder);
ky = ky(:, viewOrder);


% reshape kspace
% kspace = reshape(kspace, [Nsample, Narms_unique, Nframes, Ncoil]);
kspace = reshape(kspace, [Nsample, Nreadout_per_slice, Nframes, Ncoil]);
Nsample_k = size(kx, 1);

% reshape trajectory
% kx = reshape(kx, [Nsample_k, Narms_unique, Nframes]);
% ky = reshape(ky, [Nsample_k, Narms_unique, Nframes]);
kx = reshape(kx, [Nsample_k, Nreadout_per_slice, Nframes]);
ky = reshape(ky, [Nsample_k,Nreadout_per_slice, Nframes]);

% Trim
kspace = kspace(:,:,end-(Nframes-FramestoTrim)+1:end-FramestoTrim_end,:);

% pre-weight kspace by sqrt(DCF)
DCF = kspace_info.DCF;
kspace = kspace .* sqrt(DCF(:,1));

% Save kx, ky, kspace of all slice
kx = kx(:,:,end-(Nframes-FramestoTrim)+1:end-FramestoTrim_end);
ky = ky(:,:,end-(Nframes-FramestoTrim)+1:end-FramestoTrim_end);

kx_seq = cat(4,kx_seq,kx);
ky_seq = cat(4,ky_seq,ky);
kspace_seq = cat(5,kspace_seq,kspace);



end

% DCF 


