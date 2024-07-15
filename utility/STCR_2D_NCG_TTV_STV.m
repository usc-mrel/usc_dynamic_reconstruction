function [x_out, cost] = STCR_2D_NCG_TTV_STV(E, x0_, y, weight_tTV, weight_sTV, opt)

    if nargin < 6
        opt = setDefaultOptions();
    else
        opt = setDefaultOptions(opt);
    end
    Nmaxiter = opt.Nmaxiter;
    Nlineiter = opt.Nlineiter;
    betahow = opt.betahow;
    linesearch_how = opt.linesearch_how;
    disp(opt);
    
    scale = max(abs(x0_(:)));

    %% Regularization Operators

    % operators tfd and tv.
    T_tfd = TFD(size(x0_));
    T_tv = TV_2D(size(x0_));
    
    %{
    frame_weight = double(squeeze(max(max(abs(x0_)))));
    
    % hack the frame weight:
    frame_weight = frame_weight(1:120);
    frame_weight(1:14) = 0;
    frame_weight = frame_weight / max(vec(frame_weight));
    frame_weight = repmat(frame_weight, [nrep, 1]);
    
    T_w = T_weight(size(x0_), frame_weight);
    %}

    % --------- adjoint test on the operator TV (optional). -------------------
    test_fatrix_adjoint(T_tfd);
    test_fatrix_adjoint(T_tv);

    %% Define the L1 Approximation

    % Define potiential function as fair-l1.
    l1_func = potential_fun('fair-l1', 0.01);    % with delta  = 0.01

    %% Solver -> NCG 
    % need to define B, gradF, curvf, x0, niter, ninner, P, betahow, fun

    % -------------------------------------------------------------------------
    % prep for the NCG routine
    % -------------------------------------------------------------------------

    % Scale Regularization parameters
    lambdaTFD = weight_tTV * scale;
    lambdatTV = weight_sTV * scale;

    % ----- Data Consistency Term Related -------------------------------------
    gradDC = @(x) x - y;
    curvDC = @(x) 1;

    % ------------------ TTV Term Related -------------------------------------
    gradTFD = @(x) lambdaTFD * l1_func.dpot(x);
    curvTFD = @(x) lambdaTFD * l1_func.wpot(x);

    % ------------------ STV Term Related -------------------------------------
    gradtTV = @(x) lambdatTV * l1_func.dpot(x);
    curvtTV = @(x) lambdatTV * l1_func.wpot(x);

    % ------------------ Intermediate Step Cost -------------------------------
    costf = @(x,iter) each_iter_fun(E, T_tfd, T_tv, lambdaTFD, lambdatTV, ...
                                 l1_func, y, x, iter);

    % ------------- necessary for NCG routine ---------------------------------
    B = {E, T_tfd, T_tv};
    gradF = {gradDC, gradTFD, gradtTV};
    curvF = {curvDC, curvTFD, curvtTV};

    %% Actual Solver Here
    tic
    [x, out] = ncg(B, gradF, curvF, x0_, Nmaxiter, Nlineiter, eye, betahow, linesearch_how, costf);
    % [x, out] = ncg_inv_mm(B, gradF, curvF, x0_, Nmaxiter, Nlineiter, eye, betahow, costf);
    toc

    %% Display the Result
    img_recon = gather(x);
    x_out = (flipud(img_recon));
    cost = structArrayToStructWithArrays(cell2mat(out));
end

%% Helper Functions
function [struct] = each_iter_fun(F, T_tfd, T_tv, lambdaTFD, lambdatTV, l1_func, y, x, iter)
    % added normalization
    N = numel(x);
    struct.fidelityNorm = (0.5 * (norm(vec(F * x - y))^2)) / N;
    struct.spatialNorm = sum(vec(lambdatTV * l1_func.potk(T_tv * x))) / N;
    struct.temporalNorm = sum(vec(lambdaTFD * l1_func.potk(T_tfd * x))) / N;
    struct.totalCost = struct.fidelityNorm + struct.spatialNorm + struct.temporalNorm;
end


function options = setDefaultOptions(options)
    % Define the default options, use them if they aren't user defined.
    defaultOptions = struct('Nmaxiter', 100, ...
                            'Nlineiter', 20, ...
                            'betahow', 'DY', ...
                            'linesearch_how', 'mm');

    % If options is not provided or is empty, use default options
    if nargin < 1 || isempty(options)
        options = defaultOptions;
        return;
    end
    
    % Get the field names of the default options
    defaultFields = fieldnames(defaultOptions);
    
    % Iterate over each field in the default options
    for i = 1:numel(defaultFields)
        field = defaultFields{i};
        if ~isfield(options, field)
            options.(field) = defaultOptions.(field);
        end
    end
end
