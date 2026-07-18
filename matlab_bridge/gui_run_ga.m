function [bestX, fval, rewardTime, rewardData] = gui_run_ga(overridesJson, progressFile, stopFile, doTrain)
% GUI_RUN_GA  Bridge called from the Python GUI to run the GA humanoid walker.
%
%   [bestX, fval, rewardTime, rewardData] = ...
%       gui_run_ga(overridesJson, progressFile, stopFile, doTrain)
%
%   overridesJson : JSON string of parameter overrides (see fields below).
%   progressFile  : path to a CSV the OutputFcn appends "generation,bestFitness".
%                   The Python side polls this file to draw the live curve.
%   stopFile      : path to a flag file. If it appears mid-run, the GA halts
%                   cleanly (this is how the GUI "Stop" button works).
%   doTrain       : logical. false => use the built-in pretrained solution.
%
%   Returns the best decision vector, its fitness, and the reward time-series
%   of the best/simulated individual (rewardData is per-timestep reward; note
%   fitness = -sum(rewardData), because ga minimises).
%
%   The model only logs one signal (yout{1} = Reward). To plot torso height,
%   velocity, joint angles, etc., log those signals inside the .slx and return
%   them here in the marked slot below.

% ---- Recognised override fields (all optional) --------------------------
%   reward.forwardRewardWeight   reward.timestepRewardWeight
%   reward.powerPenaltyWeight    reward.verticalPenaltyWeight
%   reward.lateralPenaltyWeight
%   controller.hipFrontalStiffness / kneeStiffness / ankleStiffness
%   controller.hipFrontalDamping   / kneeDamping   / ankleDamping
%   nPoints  gaitPeriodMin  gaitPeriodMax
%   maxGenerations  populationSize  fitnessLimit  useParallel
% -------------------------------------------------------------------------

    if nargin < 4 || isempty(doTrain);      doTrain = false;      end
    if nargin < 3;                          stopFile = '';        end
    if nargin < 2;                          progressFile = '';    end

    ov = struct();
    if ~isempty(overridesJson)
        ov = jsondecode(overridesJson);
    end

    % Fresh progress file
    if ~isempty(progressFile)
        fid = fopen(progressFile, 'w');
        if fid > 0; fclose(fid); end
    end

    % ---- Load base parameters, then apply overrides ---------------------
    sm_humanoid_walker_ga_parameters;    % defines `params`
    params = applyOverrides(params, ov); %#ok<NODEF>

    mdlName = 'sm_humanoid_walker_ga';
    load_system(mdlName);   % must be loaded before get_param / sim

    % ---- Optimisation variable bounds -----------------------------------
    nPoints = getfielddef(ov, 'nPoints', 4);
    gaitLo  = getfielddef(ov, 'gaitPeriodMin', 0.5);
    gaitHi  = getfielddef(ov, 'gaitPeriodMax', 2.0);

    ub = ones([1, nPoints*3]);
    lb = -ub;
    ub(end+1) = gaitHi;
    lb(end+1) = gaitLo;

    % ---- GA options -----------------------------------------------------
    useParallel = logical(getfielddef(ov, 'useParallel', false));
    opts = optimoptions('ga');
    opts.MaxGenerations = getfielddef(ov, 'maxGenerations', 20);
    opts.PopulationSize = getfielddef(ov, 'populationSize', 100);
    opts.FitnessLimit   = getfielddef(ov, 'fitnessLimit', -1000);
    opts.Display        = 'iter';
    opts.UseParallel    = useParallel;
    opts.OutputFcn      = @gaOutputFcn;   % nested, captures progressFile/stopFile

    costFcn = @(x) sm_humanoid_walker_sim_walking(x, mdlName, params);

    if doTrain
        if useParallel
            % Mirror the example's speedup setup only when asked for.
            parallelFlag = true; accelFlag = true; %#ok<NASGU>
            try
                sm_humanoid_walker_ga_speedup;
            catch ME
                warning('Parallel speedup unavailable (%s). Continuing serially.', ME.message);
                opts.UseParallel = false;
            end
        end
        [bestX, fval] = ga(costFcn, length(ub), [],[],[],[], lb, ub, [], [], opts);
        bdclose(mdlName);
        if useParallel
            delete(gcp('nocreate'));
            if exist('temp','dir'); rmdir('temp','s'); end
        end
    else
        % Built-in pretrained solution from the example.
        bestX = [0.9182  0.9441  0.6705 -0.2348 -0.2725 ...
                -1.0000 -0.9120  0.5830 -0.1476  0.5914 ...
                -0.5010 -0.7179  1.3537];
        nPoints = 4;
        fval = NaN;
    end

    % ---- Simulate the chosen individual to get the reward time-series ---
    % The training branch closed the model (bdclose) during cleanup, so make
    % sure it is loaded again before touching its workspace / simulating.
    if ~bdIsLoaded(mdlName)
        load_system(mdlName);
    end

    waypoints  = sm_humanoid_walker_generate_waypoints(bestX(1:end-1), nPoints);
    gaitPeriod = bestX(end); %#ok<NASGU>

    mdlWks = get_param(mdlName, 'ModelWorkspace');
    assignin(mdlWks, 'gaitPeriod', bestX(end));
    assignin(mdlWks, 'nPoints',    nPoints);
    assignin(mdlWks, 'waypoints',  waypoints);

    simOut     = sim(mdlName, 'FastRestart', 'on', 'SrcWorkspace', 'current');
    rewardTs   = simOut.yout{1}.Values;
    rewardTime = rewardTs.Time(:);
    rewardData = rewardTs.Data(:);
    if isnan(fval)
        fval = -sum(rewardData);   % keep fitness consistent for pretrained
    end

    % ===== TIER-2 SLOT ===================================================
    % Once you log height/velocity/joint-angle signals in the .slx, read them
    % here (e.g. simOut.yout{2}.Values ...) and add them to the return list.
    % =====================================================================

    % ---- Nested streaming output function -------------------------------
    function [state, options, optchanged] = gaOutputFcn(options, state, flag) %#ok<INUSD>
        optchanged = false;
        if isempty(state.Best); best = NaN; else; best = state.Best(end); end
        if ~isempty(progressFile)
            fidp = fopen(progressFile, 'a');
            if fidp > 0
                fprintf(fidp, '%d,%.6f\n', state.Generation, best);
                fclose(fidp);
            end
        end
        % Cooperative stop from the GUI.
        if ~isempty(stopFile) && exist(stopFile, 'file')
            state.StopFlag = 'Stopped by user from GUI';
        end
    end
end

% ---- helpers ------------------------------------------------------------
function params = applyOverrides(params, ov)
    if isfield(ov, 'reward')
        f = fieldnames(ov.reward);
        for i = 1:numel(f); params.reward.(f{i}) = ov.reward.(f{i}); end
    end
    if isfield(ov, 'controller')
        f = fieldnames(ov.controller);
        for i = 1:numel(f); params.controller.(f{i}) = ov.controller.(f{i}); end
    end
    if isfield(ov, 'worldDamping'); params.simulation.worldDamping = ov.worldDamping; end
end

function v = getfielddef(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)); v = s.(name); else; v = default; end
end
