 function ob = TV_2D(image_size)
%function ob = TV_2D([mask,] args)
%|
%| Do Total variation differences
%| Inputs:
%|     image_size: vector image size [nx, ny, nframe, nc]
%|     sens: sensitivity map: size([nx, ny, coil])
%| 
%| Inspired and modified from Jeff Fessler's Gnufft object in the
%| Michigan Image Reconstruction Toolbox (MIRT).

%{
nx = image_size(1);
ny = image_size(2);
nframe = image_size(3);
%}

idim = vec(image_size)'; % [nx, ny, nframe];
odim = vec(image_size)';

forw = @(arg, x) TV_forw(x);
back = @(arg, y) TV_adj(y);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end

% Function to compute finite spatial differences along dimensions 1 and 2
function diff_result = TV_forw(input_array)
    % Compute finite differences along dimension 1 (x-axis)
    diff_x = input_array - circshift(input_array, [0, -1]);
    
    % Compute finite differences along dimension 2 (y-axis)
    diff_y = input_array - circshift(input_array, [-1, 0]);
    
    % Sum the differences along both dimensions
    diff_result = diff_x + diff_y;
end

% Function to compute the adjoint (backward) of the finite differences
function adjoint_result = TV_adj(forward_diff)
    % Compute the adjoint for dimension 1 (x-axis)
    adjoint_x = forward_diff - circshift(forward_diff, [0, 1]);
    
    % Compute the adjoint for dimension 2 (y-axis)
    adjoint_y = forward_diff - circshift(forward_diff, [1, 0]);
    
    % Sum the adjoints along both dimensions
    adjoint_result = adjoint_x + adjoint_y;
end