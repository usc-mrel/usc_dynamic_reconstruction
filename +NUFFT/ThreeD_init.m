function Data = ThreeD_init(kSpace_radial,kx,ky,para)
over_sampling = para.over_sampling;
kernel_size = para.kernel_size;

[nx,ny,nz,nof,nc] = size(kSpace_radial);
%kSpace_cart = zeros([nx*over_sampling,nx*over_sampling,nz,nof,nc],'single');
%Toeplitz_mask = zeros([nx*over_sampling,nx*over_sampling,nz,nof],'single');
Data.N = cell(1,nz-2);
Data.image = zeros(nx,nx,nz,nof,nc,'single');
for i=1:nz-2
    kSpace_temp = squeeze(kSpace_radial(:,:,i,:,:));
    kx_temp = squeeze(kx(:,:,i,:));
    ky_temp = squeeze(ky(:,:,i,:));

    mask = kSpace_temp == 0;
    mask = mask(144,:,1,1);
    
    kSpace_temp(:,mask,:,:) = [];
    kx_temp(:,mask,:,:) = [];
    ky_temp(:,mask,:,:) = [];

    Data.N{i} = NUFFT.init_new(kx_temp,ky_temp,over_sampling,kernel_size);
    Data.image(:,:,i,:,:) = NUFFT.NUFFT_adj_new(kSpace_temp,Data.N{i});

end
Data. image = ifft(Data.image,[],3);
