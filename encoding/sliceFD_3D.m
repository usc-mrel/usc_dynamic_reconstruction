 function ob = sliceFD_3D(image_size)
%function ob = TFD([mask,] args)
%|
%| Do slice finite differences (3D)
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

forw = @(arg, x) sliceFD_forw(x);
back = @(arg, y) sliceFD_adj(y);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end
 
 function [y] = sliceFD_forw(x)
    y = diff(x, 1, 3);
    y = cat(3, x(:,:,1,:) - x(:,:,end,:), y);
 end
 
function [x] = sliceFD_adj(y)
    x = -diff(y, 1, 3);
    x = cat(3, x, y(:,:,end,:) - y(:,:,1,:));
end
 