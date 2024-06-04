 function ob = sliceFD_3D(image_size, passband)
%function ob = TFD([mask,] args)
%|
%| Do slice finite differences (3D)
%| Inputs:
%|     image_size: vector image size [nx, ny, nz, nframe]
%|     passband: optional vector of which nz slices to do difference on.
%| 
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).

nx = image_size(1);
ny = image_size(2);
nz = image_size(3);
nframe = image_size(4);

if nargin < 2
    passband = 1:nz;
end

passvec = permute(zeros(nz,1), [2, 4, 1, 3]);
passvec(:,:,passband) = 1;

idim = [nx, ny, nz, nframe];
odim = [nx, ny, nz, nframe];

forw = @(arg, x) sliceFD_forw(x, passvec);
back = @(arg, y) sliceFD_adj(y, passvec);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end
 
 function [y] = sliceFD_forw(x, passvec)
    x = x .* passvec;
    y = diff(x, 1, 3);
    y = cat(3, x(:,:,1,:) - x(:,:,end,:), y);
    y = y .* passvec;
 end
 
function [x] = sliceFD_adj(y, passvec)
    y = y .* passvec;
    x = -diff(y, 1, 3);
    x = cat(3, x, y(:,:,end,:) - y(:,:,1,:));
    x = x .* passvec;
end
 