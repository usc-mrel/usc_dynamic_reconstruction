 function ob = TFD_3D(image_size)
%function ob = TFD([mask,] args)
%|
%| Do Temporal Finite Differences
%| Inputs:
%|     image_size: vector image size [nx, ny, nz, nframe, nc]
%| 
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).
arguments
    image_size (4, 1) double
end

nx = image_size(1);
ny = image_size(2);
nz = image_size(3);
nframe = image_size(4);

idim = [nx, ny, nz, nframe];
odim = [nx, ny, nz, nframe];

forw = @(arg, x) TFD_forw(x);
back = @(arg, y) TFD_adj(y);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end
 
 function [y] = TFD_forw(x)
    y = diff(x, 1, 4);
    y = cat(4, x(:,:,:,1) - x(:,:,:,end), y);
 end
 
function [x] = TFD_adj(y)
    x = -diff(y, 1, 4);
    x = cat(4, x, y(:,:,:,end) - y(:,:,:,1));
end
 