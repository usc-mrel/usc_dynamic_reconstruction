 function ob = Fnufft_2D(kx, ky, nc, matrix_size, useGPU, w, FOV_oversamp, kernel_size)
%function ob = Fnufft2D([mask,] args)
%|
%| Do fourier encoding (F) for dynamic 2D operators.
%| 
%| Inputs:
%|     kx: size(ns x nrep x nframe)
%|     ky: size(ns x nrep x nframe)
%|     nc: number of coils
%|     matrix_size: size(2 x 1), nx x ny
%|     optional arguments:
%|     w: density compensation across frame (ns x 1)
%|     FOV_oversamp: oversampling 
%|     kernel_size: size(2,1)
%|     
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).
%{
arguments
    kx (:,:,:) double
    ky (:,:,:) double
    nc double
    matrix_size (:,:) double
    useGPU double = 0
    w (:,1) double = ones(size(kx,1), 1)
    FOV_oversamp double = 1
    kernel_size (2,1) = [4,4]
end
%}

[nread, ns, nframe] = size(kx);

%% Construct the N operators for forward model calculation
matrix_size = matrix_size * FOV_oversamp;
N = NUFFT.init(kx*matrix_size(1), ky*matrix_size(2), 1, kernel_size, matrix_size(1), matrix_size(2));
N.W = sqrt(w);

if useGPU
    N.S = gpuArray(N.S);
    N.W = gpuArray(N.W);
    N.Apodizer = gpuArray(N.Apodizer);
end

idim = [matrix_size, nframe, nc];
odim = [nread, ns, nframe, nc];

forw = @(arg, x) Fnufft_forw(arg, x);
back = @(arg, y) Fnufft_adj(arg, matrix_size, y);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back, 'arg', N);
end

function y = Fnufft_forw(arg, x)
    [nx, ny, ~, ~] = size(x);
    y = NUFFT.NUFFT(x, arg) / sqrt(nx * ny);
    y = y .* arg.W;  % multiply by sqrt(dcf).
end


function x = Fnufft_adj(arg, matrix_size, y)
    x = NUFFT.NUFFT_adj(y, arg) * sqrt(matrix_size(1) * matrix_size(2));
end
