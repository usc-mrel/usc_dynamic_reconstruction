function kSpace_radial = cart2rad(kSpace_cart,N)

sx = N.siz(1);
nor = N.siz(2);
nof = N.siz(3);
nc = size(kSpace_cart,4);
nSMS = size(kSpace_cart,5);
sx_over = N.sx_over;
core_size = N.core_size;

if mod(core_size(1),2) == 0
    kSpace_cart = reshape(kSpace_cart,[(sx_over+core_size(1))*(sx_over+core_size(2)),nof,nc,nSMS]);
else
    kSpace_cart = reshape(kSpace_cart,[(sx_over+core_size(1)+1)*(sx_over+core_size(2)+1),nof,nc,nSMS]);
end
kSpace_cart = permute(kSpace_cart,[1 3 2 4]);

kSpace_radial = zeros([sx*nor*prod(core_size),nof,nc,nSMS]);
for j=1:nSMS
for i=1:nof
    kSpace_radial(:,i,:,j) = single(N.rad2cart{i}.'*double(kSpace_cart(:,:,i,j)));
end
end
kSpace_radial = reshape(kSpace_radial,[sx*nor,prod(core_size),nof,nc,nSMS]);
kSpace_radial = permute(kSpace_radial,[1 3 2 4 5]);
%kSpace_radial = reshape(kSpace_radial,[sx*nor*nof,1,nc,prod(core_size)]);
%G_all = N.Dict_c2r(N.indx_c2r,:,:);
%G_all = reshape(G_all,[sx*nor*nof,prod(core_size),nc,nc]);
%G_all = permute(G_all,[1,3,4,2]);
%kSpace_radial = bsxfun(@times,G_all,kSpace_radial);
%kSpace_radial = squeeze(sum(kSpace_radial,3));

%kSpace_radial = sum(kSpace_radial,3)/prod(core_size);

%weight_c2r = reshape(N.weight_kb,[sx*nor*nof,1,prod(core_size)]);
kSpace_radial = bsxfun(@times,kSpace_radial,N.weight_kb);
%kSpace_radial = bsxfun(@times,kSpace_radial,conj(N.phase));
kSpace_radial = sum(kSpace_radial,3);
%{
weight_c2r = reshape(G.weight1,[sx*nor*nof,1,prod(core_size)]);
kSpace_radial = bsxfun(@times,kSpace_radial,weight_c2r);
kSpace_radial = sum(kSpace_radial,3);
weight_c2r = sum(weight_c2r,3);
kSpace_radial = bsxfun(@rdivide,kSpace_radial,weight_c2r);
%}
kSpace_radial = reshape(kSpace_radial,[N.siz,nc,nSMS]);