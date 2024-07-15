function [x, out] = ncg_inv_mm(B, gradf, curvf, x0, niter, ninner, P, betahow, fun)
    % Check for missing optional arguments
    if nargin < 9
        fun = @(x, iter) [];
    end
    if nargin < 8
        betahow = 'dai_yuan'; % not actually used right now. it is always dai-yuan
    end
    if nargin < 7
        P = eye(size(x0));
    end
    if nargin < 6
        ninner = 5;
    end
    if nargin < 5
        niter = 50;
    end

    J = length(B);
    x = x0;

    out = cell(1, niter + 1);
    out{1} = fun(x0, 1);

    % Legacy variables
    dir = [];
    grad_old = [];
    grad_new = [];
    alf = 0;

    for iter = 1:niter
        Bx = cell(1, J);
        for j = 1:J
            Bx{j} = B{j} * x;
        end

        grad_new = zeros(size(x));
        for j = 1:J
            grad_new = grad_new + B{j}' * gradf{j}(Bx{j});
        end

        npgrad = -P * grad_new;

        if iter == 1
            dir = npgrad;
        else
            gamma = (vec(grad_new)' * (P * vec(grad_new))) / (vec(grad_new - grad_old)' * vec(dir));
            dir = npgrad + (gamma * dir);
        end

        Bd = cell(1, J);
        for j = 1:J
            Bd{j} = B{j} * dir;
        end

        grad_old = grad_new;

        c1f = @(alf) sum(arrayfun(@(j) vec(real(Bd{j}))' * vec(gradf{j}(Bx{j} + alf * Bd{j})), 1:J));
        
        c2f = @(alf) sum(arrayfun(@(j) vec(real(Bd{j}))' * (curvf{j}(vec(Bx{j} + alf * Bd{j})) .* vec(Bd{j})), 1:J));
        
        alf = 0;

        for in_iter = 1:ninner
            c1_alf = c1f(alf);
            c2_alf = c2f(alf);
            alf = alf - (c1_alf / c2_alf);
        end

        x = x + (alf * dir);

        for j = 1:J
            Bx{j} = Bx{j} + alf * Bd{j};
        end
        
        if mod(iter,10) == 0 || iter == niter
            fprintf(sprintf('%10.0f %10.2f %10.4f \n',iter, out{iter}.totalCost, alf));
        end

        out{iter + 1} = fun(x, iter+1);
    end
end