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
odim = [vec(image_size); 2]';

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
    diff_result = cat(length(size(input_array))+1, (diff_x),  (diff_y));
end

% Function to compute the adjoint (backward) of the finite differences
function adjoint_result = TV_adj(forward_diff)
    % Compute the adjoint for dimension 1 (x-axis)
    adjoint_x = indexLastDim(forward_diff, 1) - circshift(indexLastDim(forward_diff, 1), [0, 1]);
    
    % Compute the adjoint for dimension 2 (y-axis)
    adjoint_y = indexLastDim(forward_diff, 2) - circshift(indexLastDim(forward_diff, 2), [1, 0]);
    
    % Sum the adjoints along both dimensions
    adjoint_result = (adjoint_x) + (adjoint_y);
end


function output = indexLastDim(x, index)
    % Reshape x into a 2D matrix where the last dimension is unfolded
    % into the columns, and all other dimensions are merged into the rows.
    sz = size(x); % Get the size of the original array
    lastDim = sz(end); % The size of the last dimension
    otherDims = prod(sz(1:end-1)); % The product of the other dimensions

    % Reshape x considering it might have more than 2 dimensions. The new shape
    % has all the elements except the last dimension in the first dimension,
    % and the last dimension in the second dimension.
    reshapedX = reshape(x, [otherDims, lastDim]);

    % Now, index the first element of the last dimension.
    % Since reshapedX is a 2D matrix, we want the first column.
    output = reshapedX(:, index);

    % If needed, reshape back to original dimensions minus the last dimension
    output = reshape(output, sz(1:end-1));
end