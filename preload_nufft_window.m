NF = min(lcm(GA_steps,Narms_recon)/Narms_recon, floor(Nt/Narms_recon));
E_load = cell(NF,1);
EhE_load = cell(NF,1);
kspace_corners_mask = circular_mask_toeplitz(2*oversampling*matrix_size(1));
Z = Zeropad_2D([oversampling*matrix_size,1,Ncoil],2);
Fou = FFT2_fatrix([2*oversampling*matrix_size,1,Ncoil]);
D = oversampling*matrix_size(1);
for i = 1:1:NF % Preloads NUFFT operators
    ti = 1+Narms_recon*(i-1);
    tf = ti-1+Narms_window;
    kx_fr = reshape(kx(:,:,ti:tf),[Nsample,Narms_window,1]);
    ky_fr = reshape(ky(:,:,ti:tf),[Nsample,Narms_window,1]);
    F = Fnufft_2D(kx_fr, ky_fr, Ncoil, matrix_size, useGPU, DCF(:,1), oversampling, [4,4]);
    lin_phase = -2*pi*(kx_fr+ky_fr)*(-D);
    W = repmat(DCF(:,1:Narms_window),[1,1,1,Ncoil]);
    F2N = Fnufft_2D(kx_fr, ky_fr, Ncoil, 2*matrix_size, useGPU, ones(size(DCF(:,1))), oversampling, [8,8]);
    psf0 = fftshift2(F2N'*W);
    kxpsf = repmat((-D:D-1)',[1,2*D]); 
    kypsf = repmat(-D:D-1,[2*D,1]);
    psf = myifft2(myfft2(psf0).*kspace_corners_mask.*exp(-1i*pi*(kxpsf+kypsf)/(2*D)));
    M = elementwise(fft2(psf)/sqrt(prod(matrix_size)));
    E_load{i} = F*C;
    if toeplitz_flag
        EhE_load{i} = C'*Z'*Fou'*M*Fou*Z*C;
    else
        
        EhE_load{i} = C'*F'*F*C;
    end
end


function y = myfft2(x)
y = fftshift(fftshift(fft2(x),1),2);
end

function y = myifft2(x)
y = ifft2(ifftshift(ifftshift(x,1),2));
end

