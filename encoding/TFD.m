 function ob = TFD(image_size)
%function ob = TFD([mask,] args)
%|
%| Do Temporal Finite Differences
%| Inputs:
%|     image_size: vector image size [nx, ny, nframe, nc]
%|     sens: sensitivity map: size([nx, ny, coil])
%| 
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).
arguments
    image_size (3, 1) double
end

nx = image_size(1);
ny = image_size(2);
nframe = image_size(3);

idim = [nx, ny, nframe];
odim = [nx, ny, nframe];

forw = @(arg, x) TFD_forw(x);
back = @(arg, y) TFD_adj(y);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end
 
 function [y] = TFD_forw(x)
    y = diff(x, 1, 3);
    y = cat(3, x(:,:,1) - x(:,:,end), y);
 end
 
function [x] = TFD_adj(y)
    x = -diff(y, 1, 3);
    x = cat(3, x, y(:,:,end) - y(:,:,1));
end
 