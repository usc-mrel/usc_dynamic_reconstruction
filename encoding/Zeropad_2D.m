 function ob = Zeropad_2D(image_size, pad_ratio)
%function ob = TFD([mask,] args)
%|
%| Do Temporal Finite Differences
%| Inputs:
%|     image_size: vector image size [nx, ny, nframe, nc]
%|     sens: sensitivity map: size([nx, ny, coil])
%| 
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).
% arguments
%     pad_ratio
%     image_size 
% end


idim = image_size;
odim = image_size;
odim(1:2) = odim(1,2)*pad_ratio;

forw = @(arg, x) Z_forw(x, pad_ratio);
back = @(arg, y) Z_adj(y, pad_ratio);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end
 
 function [y] = Z_forw(x,pad_ratio)
    
    Nx = size(x,1);
    Ny = size(x,2);
    
%     tic
%     y = gpuArray(zeros([Nx*pad_ratio,Ny*pad_ratio,size(x,3:ndims(x))]));
%     toc
%     tic
%     y(ceil(Nx*pad_ratio/2)-floor(Nx/2)+(1:Nx),ceil(Ny*pad_ratio/2)-floor(Ny/2)+(1:Ny),:)=x;
%     toc

    y = padarray(x,round((pad_ratio-1)/2*[Nx,Ny]),0,'both');

 end
 
function [x] = Z_adj(y, pad_ratio)
    Nx = size(y,1);
    Ny = size(y,2);
    x = y(round(Nx/2-Nx/pad_ratio/2)+(1:round(Nx/pad_ratio)), round(Ny/2-Ny/pad_ratio/2)+(1:round(Ny/pad_ratio)), : );
    odim = size(y);
    odim(1:2) = round(odim(1:2)/pad_ratio);
    x = reshape(x,odim);
end
 