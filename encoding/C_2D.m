 function ob = C_2D(image_size, sens, useGPU)
%function ob = C_2d([mask,] args)
%|
%| Do coil map encoding (C).
%| Inputs:
%|     image_size: vector image size [nx, ny, nframe, nc]
%|     sens: sensitivity map: size([nx, ny, nframe (or 1 if static), coil])
%| 
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).
%{
arguments
    image_size (4, 1) double
    sens (:,:,:, :) double
    useGPU double = 0
end
%}

nx = image_size(1);
ny = image_size(2);
nframe = image_size(3);
nc = image_size(4);

idim = [nx, ny, nframe];
odim = [nx, ny, nframe, nc];

forw = @(arg, x) C_forw(arg, x);
back = @(arg, y) C_adj(arg, y);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back, 'arg', sens);

 end

 
 function [y] = C_forw(sens, x)
    % expand the multi-coil image.
    y = x;
    nc = size(sens, 4);
    for i = 1:nc
       y(:,:,:,i) = x .* sens(:,:,:,i); 
    end
 end
 
 function [x] = C_adj(sens, y)
    % combine the image into coils
    x = squeeze(sum(y .* conj(sens), 4));
 end
 