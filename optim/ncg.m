function [x, out, intx] = ncg(B, gradf, curvf, x0, niter, ninner, P, betahow, linesearch_how, fun)
    
    % Check for missing optional arguments
    if nargin < 10
        fun = @(x, iter) [];
    end
    
    if nargin < 9
        linesearch_how = 'mm';
    end
    
    if nargin < 8
        betahow = 'FR'; % See options below
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
    
    % ---------------------------------------------------------------------
    fprintf([repmat('-', [1, 75]), '\n'])
    disp('begin nonlinear conjugate gradient reconstruction...');
    fprintf('  Iter \t Total Cost \t Step Size \t Beta \n');
    fprintf([repmat('-', [1, 75]), '\n'])
    % ---------------------------------------------------------------------
    
    J = length(B);              % Number of terms in the cost function
    x = x0;                     % Initial Point

    out = cell(1, niter + 1);   % Cost Term is stored here
    out{1} = fun(x0, 1);        % Cost associated with the initial estimate

    % Legacy variables
    s = [];                         % Update Direction
    delta_x_old = [];               % Gradient Term at iter-1
    delta_x = [];                   % Gradient Term at iter
    alpha = 0;                      % Step Size
    
    % ---------------------------------------------------------------------
    for iter = 1:niter 
        
        % --------------- Calculate Gradients -----------------------------
        Bx = cell(1, J);
        for j = 1:J
            Bx{j} = B{j} * x;
        end

        nabla_x = zeros(size(x));
        for j = 1:J
            nabla_x = nabla_x + B{j}' * gradf{j}(Bx{j});
        end

        delta_x = -P * nabla_x;
        
        % ------------------ Find Update Direction ------------------------
        if iter == 1
            s = delta_x;
        else
            beta = calculate_beta(delta_x, delta_x_old, s, betahow, 0);
            s = delta_x + (beta * s);
        end

        Bd = cell(1, J);
        for j = 1:J
            Bd{j} = B{j} * s;
        end

        delta_x_old = delta_x;
        
        % ---------------------- Step Size --------------------------------
        alpha = line_search(Bd, Bx, gradf, curvf, ninner, J, fun, x, s, linesearch_how);

        % --------------------- Update solution ---------------------------
        x = x + (alpha * s);

        for j = 1:J
            Bx{j} = Bx{j} + alpha * Bd{j};
        end
        
        % -----------------------------------------------------------------
        % Print out a summary
        if mod(iter,10) == 0 || iter == niter
            fprintf(sprintf('%10.0f %10.2f %10.4f %10.4f \n',iter, out{iter}.totalCost, alpha, beta));
        end
        % -----------------------------------------------------------------
        
        
        % Record the Cost function from this iteration 
        out{iter + 1} = fun(x, iter+1);
        
        % -----------------------------------------------------------------
        % Another Stopping Criteria
        if iter > 1
            if alpha < 1e-5
                break;
            end
        end
        % -----------------------------------------------------------------
    end
end

%% ------------------------------------------------------------------------
%% Calculate beta 
function beta = calculate_beta(delta_x, delta_x_old, s, betahow, reset_flag)

epsilon = 1e-6*max(delta_x(:));

switch betahow
    case 'FR'       % Fletcher-Reeves
        beta = (delta_x(:)' * delta_x(:)) / (delta_x_old(:)' * delta_x_old(:) + epsilon);
    case 'PR'       % Polak-Ribiere
        beta = delta_x(:)' * (delta_x(:) - delta_x_old(:)) / (delta_x_old(:)' * delta_x_old(:) + epsilon);
    case 'HS'       % Hestenes-Stiefel
        beta = delta_x(:)' * (delta_x(:) - delta_x_old(:)) / (-s(:)' * (delta_x(:) - delta_x_old(:)) + epsilon);
    case 'DY'       % Dai-Yuan
        beta = delta_x(:)' * delta_x(:) / (-s(:)' * (delta_x(:) - delta_x_old(:)) + epsilon);
    case 'YT'
        beta = delta_x(:)' * delta_x(:) / (s(:)' * s(:));
    case 'GD'       % Gradient Descent
        beta = 0;
end

if reset_flag
    beta = max(0, beta);
end

end
%%
