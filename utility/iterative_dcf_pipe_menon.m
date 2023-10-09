function [dcf] = iterative_dcf_pipe_menon(F)
    % this function is not complete. 

    w = ones(F.odim);
    for i = 1:100
        ffh = F * F' * w;
        w = w ./ abs(ffh);
        resid = max(abs(ffh(:) - 1))
    end
    dcf = w;
end