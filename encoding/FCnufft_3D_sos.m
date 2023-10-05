 function ob = FCnufft_3D_sos(kx_3d, ky_3d, matrix_size, sens, w, mask)
%function ob = FCnufft_sos([mask,] args)
%|
%| Do sensitivity map encoding (C) and fourier encoding (F) in one step.
%| Constructed as a fatrix2 object. The reason for doing jointly is to save
%| memory, whereas doing them separately incurs a memory cost along the coil
%| dimension.
%| 
%| Construct Gnufft object that computes nonunform samples of the FT
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
%|     matrix_size: size(3 x 1), nx x ny x nz
%|     sens: size(5 x 1), nx x ny x nz x 1 x nc
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

        N.S = gpuArray(N.S);
        N.Apodizer = gpuArray(N.Apodizer);
        N.W = gpuArray(N.W);
        
        N_all(j, i) = N;
    end
end

idim = [matrix_size, nframe];
odim = [nread, ns,nkz,nframe,size(sens,5)];

forw = @(arg, x) FCnufft_sos_forw(arg, kx_3d, ky_3d, sens, x, mask);
back = @(arg, y) FCnufft_sos_adj(arg, matrix_size, sens, y, mask);
gram = @(ob, W, reuse) FCnufft_sos_gram(N_all, sens, kx_3d, ky_3d, mask, ob);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back, 'arg', N_all, ...
    'gram', gram);
end

function y = FCnufft_sos_forw(arg, kx_3d, ky_3d, sens, x, mask)

    [nx, ny, nz, nframe] = size(x);
    nc = size(sens,5);

    y = zeros([size(kx_3d), nc], class(x));
    
    
    for i = 1:nc
        fidelity_update = bsxfun(@times,x,sens(:,:,:,:,i));
        fidelity_update = fftshift(fidelity_update, 3);
        fidelity_update = fft(fidelity_update,[],3) / sqrt(nz);
        fidelity_update = fftshift(fidelity_update, 3);   
        for islice = 1:size(fidelity_update, 3)
            for ispiral = 1:size(y, 2)
                y(:,ispiral,islice,:, nc) = NUFFT.NUFFT(permute(fidelity_update(:,:,islice,:),[1,2,4,3]),arg(ispiral,islice)) / (sqrt(nx*ny));
                y = y .* arg(ispiral,islice).W;
            end
        end
    end
    y = (y .* mask);
end


function x = FCnufft_sos_adj(arg, matrix_size, sens, y, mask, w)
    [nread, ns, kz_steps, nframe, nc] = size(y);
   
    
    x = zeros([matrix_size, nframe], class(y));
    y = y .* mask;
   

    for i = 1:nc
        y_c = y(:,:,:,:,i);
        x_c = zeros(size(x), class(x));
        
        for islice = 1:size(x, 3)
            fidelity_update_temp = zeros([arg(1).size_image, arg(1).size_data(3)], class(x));
            for ispiral = 1:size(y_c, 2)
                fidelity_update_temp = fidelity_update_temp + (NUFFT.NUFFT_adj(permute(y_c(:,ispiral,islice,:),[1,2,4,3]),arg(ispiral, islice)) * sqrt(matrix_size(1) * matrix_size(2)));
            end
            x_c(:,:,islice,:) = fidelity_update_temp;
        end

        x_c = fftshift(x_c, 3);
        x_c = ifft(x_c,[],3) * sqrt(matrix_size(3));
        x_c = fftshift(x_c,3);

        x_c = bsxfun(@times,x_c,conj(sens(:,:,:,:,i)));
        x = x + x_c;
    end
end


function [x,norm] = FCnufft_sos_gram(arg, sens, kx_3d, ky_3d, mask, x1)
    [nx, ny, nz, nframe] = size(x1);
    nc = size(sens,5);
    
    fidelity_update_all = zeros(size(x1),class(x1));
    fidelity_norm = 0;
    for i = 1:nc

        fidelity_update = bsxfun(@times,x1,sens(:,:,:,:,i));
        fidelity_update = fftshift(fidelity_update, 3);
        fidelity_update = fft(fidelity_update,[],3);
        fidelity_update = fftshift(fidelity_update, 3);
        kSpace_spiral = zeros(size(kx_3d), class(x1));
        kSpace_spiral = kSpace_spiral(:,:,:,:,1);

        for islice = 1:size(fidelity_update, 3)
            for ispiral = 1:size(kSpace_spiral, 2)
                kSpace_spiral(:,ispiral,islice,:) = NUFFT.NUFFT(permute(fidelity_update(:,:,islice,:),[1,2,4,3]),arg(ispiral,islice));
            end
        end

        % kSpace_spiral = Data.kSpace(:,:,:,:,i) - kSpace_spiral;
        kSpace_spiral = kSpace_spiral .* mask;
        fidelity_norm = fidelity_norm + sum(abs(vec(kSpace_spiral .* (arg(1).W).^0.5)).^2) / prod(arg(1).size_kspace) / size(fidelity_update, 3);
        for islice = 1:size(fidelity_update, 3)
            fidelity_update_temp = zeros([arg(1).size_image, arg(1).size_data(3)], class(fidelity_update));
            for ispiral = 1:size(kSpace_spiral, 2)
                fidelity_update_temp = fidelity_update_temp + NUFFT.NUFFT_adj(permute(kSpace_spiral(:,ispiral,islice,:),[1,2,4,3]),arg(ispiral, islice));
            end
            fidelity_update(:,:,islice,:) = fidelity_update_temp;
        end

        fidelity_update = fftshift(fidelity_update, 3);
        fidelity_update = ifft(fidelity_update,[],3);
        fidelity_update = fftshift(fidelity_update,3);

        
        fidelity_update = bsxfun(@times,fidelity_update,conj(sens(:,:,:,:,i)));
        fidelity_update_all = fidelity_update_all + fidelity_update;
    end
    norm   = sqrt(fidelity_norm);
    x = fidelity_update_all;
end