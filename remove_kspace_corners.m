function y = remove_kspace_corners(x)
y = ifft2(ifftshift(fftshift(fft2(x)).*circular_mask(size(x,1))));
end

