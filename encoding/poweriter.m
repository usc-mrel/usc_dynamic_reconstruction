function [max_eig, spect_norm] = poweriter(E, niter)
    
    bk = gpuArray(randn(E.idim));
    spect_norm = [];
    
    for i = 1:niter
       fprintf("starting next iter\n");
       bk = E' * (E * bk);
       bk = bk / norm(bk(:));
       
       spect_norm(i) = gather((conj(bk(:))' * vec(E' * E * bk)) / (conj(bk(:))' * bk(:)));
       fprintf("spectral norm: %d\n", spect_norm(i));
    end
    
    max_eig = bk;

end