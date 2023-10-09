% 3D STCR recon using a "Fatrix" operator.
ccc

addpath('encoding/');
addpath('optim');

all_mat = dir('/server/home/pkumar/mri_data/disc/heart/vol0721_20231004/raw_hawk/*pk*.mat');
FOV_recon = [640, 640];
nfile = length(all_mat);

for ii = [5] %[1:nfile]
file_name = fullfile(all_mat(ii).folder, all_mat(ii).name);
load(file_name)

% trim the TRs:
TRTrim = 150;
% find the first instance of zero after the first series of zero.
kspace_info.viewOrder(:, 1:TRTrim) = [];
kspace_info.RFIndex(:, 1:TRTrim) = [];
kspace(:, 1:TRTrim, :) = [];
kspace_info.timeStamp(1:TRTrim) = [];

para.kspace_info = kspace_info;

nos_one = kspace_info.user_nKzEncoding;

res = [kspace_info.user_ResolutionX, kspace_info.user_ResolutionY];
matrix_size = round(FOV_recon ./ res / 2) * 2;
matrix_size = [matrix_size, kspace_info.user_nKzEncoding];

sz = matrix_size(3);

sx  = size(kspace, 1);
nos = size(kspace, 2);
nc  = size(kspace, 3);

kx = kspace_info.kx_GIRF * matrix_size(1);
ky = kspace_info.ky_GIRF * matrix_size(1);
kz = kspace_info.RFIndex + 1;

nframe = nos / nos_one;
nframe = floor(nframe);

nos         = nframe * nos_one;
kspace      = kspace(:, 1:nos, :);
view_order  = kspace_info.viewOrder(1:nos);
kx          = kx(:, view_order);
ky          = ky(:, view_order);

    
%% pre-allocate kSpace, kx, ky, w
kspace_3d = zeros(sx, 1, sz, nframe, nc, 'single');
kx_3d = zeros(sx, 1, sz, nframe);
ky_3d = zeros(sx, 1, sz, nframe);

% put the spirals at correcte location. This should also work for random
% order.
for i = 1:nos
    slice_idx = kz(i);
    frame_idx = ceil(i/nos_one);
    
    kSpace_temp = kspace(:,i,:);
    kx_temp = kx(:,i,:);
    ky_temp = ky(:,i,:);
    
    ns = sum(kspace_3d(1,:,slice_idx,frame_idx,1)~=0)+1;

    kspace_3d(:,ns,slice_idx,frame_idx,:) = kSpace_temp;
    kx_3d(:,ns,slice_idx,frame_idx,:) = kx_temp;
    ky_3d(:,ns,slice_idx,frame_idx,:) = ky_temp;
end

w = kspace_info.DCF(:,1);

% get the sensitivity maps. This uses CPU and is therefore slow.
% It is recommended you save the sens map in a separate file and just load.
calcSmap = 1;
mask = kspace_3d(1, :, :, :, 1) ~= 0;
if (calcSmap)
    F = Fnufft_3D_sos(kx_3d, ky_3d, nc, matrix_size, w, mask);
    image = F' * kspace_3d;
    sens = get_sens_map(image, '3D');
    
    % ADJOINT TEST (optional)
    % test_fatrix_adjoint(F);

    save('sens_map.mat', 'sens')
else
    load('sens_map.mat');
end

% Unbalanced DCF means no multiplication by sqrt(w).
% pre weight the kSpace data fidelity by sqrt(DCF).
rootw = sqrt(w);
kspace_3d = kspace_3d .* rootw;

% generate the encoding operator used for NCG. 
% This uses GPU and therefore is faster than CPU.
E = FCnufft_3D_sos(kx_3d, ky_3d, matrix_size, sens, rootw, mask);

% optional adjoint test.
% test_fatrix_adjoint(E);

% generate first estimate m0
x0 = E' * kspace_3d;


%% to run NCG, need to define B, gradF, curvf, x0, niter, ninner, P, betahow, fun
% Define potiential function as fair-l1.
l1_func = potential_fun('fair-l1', 0.2);

% operators tfd and tv.
T_tfd = TFD_3D(size(x0));
T_tv = TV_2D(size(x0));
T_slice_fd = sliceFD_3D(size(x0));

% test adjoint (optional)
% test_fatrix_adjoint(T_tfd);
% test_fatrix_adjoint(T_tv);
% test_fatrix_adjoint(T_slice_fd);


%% parameter sweep

for lTFD = [0.001 0.002 0.005 0.01]
    for lTV = [0.001 0.002 0.005]
        for lsTV = [0.001 0.01 0.1]
            % define regularization paramters
            scale = max(abs(x0(:)));
            lambdaTFD = lTFD * scale;
            lambdatTV = lTV * scale;
            lambdasFD = lsTV * scale;
            
            filename = sprintf('./recon_data/new_ncg/%s_narm_%g_ttv_%04g_stv_%04g_sstv%04g_560_fov_20230428.mat', all_mat(ii).name(1:end-4), nos_one, lambdaTFD / scale, lambdatTV / scale, lambdasFD/scale);
            
            if isfile(filename)
                continue
            end

            gradDC = @(x) x - kspace_3d;
            curvDC = @(x) 1;

            gradTFD = @(x) lambdaTFD * l1_func.dpot(x);
            gradtTV = @(x) lambdatTV * l1_func.dpot(x);
            gradsFD = @(x) lambdasFD * l1_func.dpot(x);

            curvTFD = @(x) lambdaTFD * l1_func.wpot(x);
            curvtTV = @(x) lambdatTV * l1_func.wpot(x);
            curvsFD = @(x) lambdasFD * l1_func.wpot(x);

            costf = @(x,y) each_iter_fun(E,T_tfd,T_tv, T_slice_fd, lambdaTFD,lambdatTV, lambdasFD, l1_func,kspace_3d,x,y);

            B = {E, T_tfd, T_tv};
            gradF = {gradDC, gradTFD, gradtTV, gradsFD};
            curvF = {curvDC, curvTFD, curvtTV, curvsFD};

            x0 = gpuArray(x0);

            tic
            [x, out] = ncg_inv_mm(B, gradF, curvF, x0, 300, 20, eye,'dai-yuan', costf);
            toc

            image_recon = gather(x);
            image_recon = imrotate(fliplr(image_recon), 90);

            out = cell2mat(out);
            Cost = structArrayToStructWithArrays(out);
            plotCost(Cost);
            para.Cost = Cost;

            % PK 2023.10.08: don't save for now can easily uncomment.
            % save(filename, 'image_recon', 'para')
        end
    end
end

end

function [struct] = each_iter_fun(E, T_tfd, T_tv, T_slice_fd, lambdaTFD, lambdatTV, lambdasFD, l1_func, kspace, x, y)
    struct.fidelityNorm = (0.5 * (norm(vec(E * x - kspace))^2));
    struct.spatialNorm = sum(vec(lambdatTV * l1_func.potk(T_tv * x)));
    struct.temporalNorm = sum(vec(lambdaTFD * l1_func.potk(T_tfd * x)));
    struct.sliceNorm = sum(vec(lambdasFD * l1_func.potk(T_slice_fd * x)));
    struct.totalCost = struct.fidelityNorm + struct.spatialNorm + struct.temporalNorm;
    fprintf(" ------ iteration %i, cost: %d -------- \n", y, struct.totalCost);
end

