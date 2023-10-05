 function ob = Fnufft_3D_sos(kx_3d, ky_3d, nc, matrix_size, w, mask)
%function ob = Fnufft_sos([mask,] args)
%|
%| Do fourier encoding (F).
%| 
%| Construct object that computes nonunform samples of the FT
%| of signals with dimensions [(Nd)] approximately via the NUFFT.
%| Expects a z direction which does computation along the z direction.
%| This function only exists as a bridge to allow "matrix-like" operations
%| to Ye's NUFFT operations.
%| 
%| 
%| For basic testing, I will restrict parameters as input. 
%| To modify things like oversampling/etc just edit this file directly.
%|
%| Inputs:
%|     kx_3d: size(ns x nrep x nkz x nframe)
%|     ky_3d: size(ns x nrep x nkz x nframe)
%|     nc: number of coils
%|     matrix_size: size(3 x 1), nx x ny x nz
%|     w: density compensation across frame (ns x 1)
%|     mask: size(1 x nrep x nkz x nframe)
%|
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).

[nread, ns, nkz, nframe] = size(kx_3d);

%% Construct the N operators for forward model calculation...
for i = 1:nkz
    for j = 1:size(kx_3d, 2)
        kx_temp = permute(kx_3d(:,j,i,:),[1,2,4,3]);
        ky_temp = permute(ky_3d(:,j,i,:),[1,2,4,3]);
        N = NUFFT.init(kx_temp, ky_temp, 1, [4,4], matrix_size(1), matrix_size(1));
        N.W = w;
        N_all(j, i) = N;
    end
end

idim = [matrix_size, nframe, nc];
odim = [nread, ns, nkz, nframe, nc];

forw = @(arg, x) Fnufft_sos_forw(arg, kx_3d, ky_3d, x, mask);
back = @(arg, y) Fnufft_sos_adj(arg, matrix_size, y, mask);
gram = @(ob, W, reuse) Fnufft_sos_gram(N_all, kx_3d, ky_3d, mask, ob);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back, 'arg', N_all);
end

function y = Fnufft_sos_forw(arg, kx_3d, ky_3d, x, mask)

    [nx, ny, nz, nframe, nc] = size(x);
    
    y = zeros([size(kx_3d), nc], class(x));

    x = fftshift(x, 3);
    x = fft(x,[],3) / sqrt(nz);
    x = fftshift(x, 3);   
    for islice = 1:size(x, 3)
        for ispiral = 1:size(y, 2)
            y(:,ispiral,islice,:, :) = NUFFT.NUFFT(permute(x(:,:,islice,:, :),[1,2,4,5, 3]),arg(ispiral,islice)) / (sqrt(nx*ny));
        end
    end
    y = y .* arg(1).W;
    y = y .* mask;
end


function x = Fnufft_sos_adj(arg, matrix_size, y, mask)
    [nread, ns, kz_steps, nframe, nc] = size(y);
    
    x = zeros([matrix_size, nframe, nc], class(y));
    y = y .* mask;
    
    for islice = 1:size(x, 3)
        fidelity_update_temp = zeros([arg(1).size_image, arg(1).size_data(3), nc], class(x));
        for ispiral = 1:size(y, 2)
            fidelity_update_temp = fidelity_update_temp + (NUFFT.NUFFT_adj(permute(y(:,ispiral,islice,:,:),[1,2,4,5,3]), arg(ispiral,islice)) * sqrt(matrix_size(1) * matrix_size(2)));
        end
        x(:,:,islice,:,:) = fidelity_update_temp;
    end
    
    x = fftshift(x, 3);
    x = ifft(x,[],3) * sqrt(kz_steps);
    x = fftshift(x,3);
end
