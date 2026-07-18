function [episodeReward, rewardTime, rewardData] = gui_run_rl(overridesJson, doTrain, useParallel, agentFile)
% GUI_RUN_RL  Bridge called from the Python GUI to run the DDPG humanoid walker.
%
%   [episodeReward, rewardTime, rewardData] = ...
%       gui_run_rl(overridesJson, doTrain, useParallel, agentFile)
%
%   overridesJson : JSON string of parameter overrides (same reward/controller
%                   fields as the GA bridge, plus maxEpisodes / stopReward /
%                   saveReward).
%   doTrain       : logical. false => load a pretrained agent (see agentFile).
%   useParallel   : logical. Requires Parallel Computing Toolbox if true.
%   agentFile     : (optional) path to a .mat holding a trained agent. If given
%                   and doTrain is false, the agent is auto-detected by class
%                   (rl.agent.*) regardless of its variable name, and any
%                   training results in the file are returned as episodeReward.
%                   If empty, the shipped sm_humanoid_walker_saved_agent is used.
%
%   Live progress for RL is shown in MATLAB's own Episode Manager (the
%   train() call opens it). This function returns the full episode-reward
%   vector when training finishes, plus the reward time-series of a validation
%   rollout, which the GUI plots.
%
%   NOTE: full training takes hours. Prefer doTrain=false for quick demos.

    if nargin < 4; agentFile = ''; end

    if nargin < 3 || isempty(useParallel); useParallel = false; end
    if nargin < 2 || isempty(doTrain);     doTrain     = false; end

    ov = struct();
    if ~isempty(overridesJson); ov = jsondecode(overridesJson); end

    % ---- Parameters -----------------------------------------------------
    sm_humanoid_walker_rl_parameters;    % defines `params`
    params = applyOverrides(params, ov); %#ok<NODEF>

    mdlName = 'sm_humanoid_walker_rl';
    load_system(mdlName);

    % ---- Environment (fixed spec from the example) ----------------------
    numAct          = 6;                 % 6 leg joints
    actionInfo      = rlNumericSpec([numAct 1], 'LowerLimit', -1, 'UpperLimit', 1);
    actionInfo.Name = 'jointDemands';

    numObs               = 25 + 6;       % 25 sensors + 6 previous actions
    observationInfo      = rlNumericSpec([numObs 1]);
    observationInfo.Name = 'observations';

    blk = [mdlName, '/RL Agent'];
    env = rlSimulinkEnv(mdlName, blk, observationInfo, actionInfo);

    % ---- Actor / critic networks ---------------------------------------
    actorLayerSizes  = [400 300];
    criticLayerSizes = [400 300];
    sm_humanoid_walker_create_networks;  % script: builds `actor`, `critic`

    criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3, ...
                        'GradientThreshold',1,'L2RegularizationFactor',2e-4);
    actorOptions  = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-4, ...
                        'GradientThreshold',1,'L2RegularizationFactor',1e-5); %#ok<NASGU>

    agentOptions                                     = rlDDPGAgentOptions;
    agentOptions.SampleTime                          = params.simulation.Ts;
    agentOptions.DiscountFactor                      = 0.99;
    agentOptions.MiniBatchSize                       = 128;
    agentOptions.ExperienceBufferLength              = 1e6;
    agentOptions.TargetSmoothFactor                  = 1e-3;
    agentOptions.NoiseOptions.MeanAttractionConstant = 5;
    agentOptions.NoiseOptions.Variance               = 0.4;
    agentOptions.NoiseOptions.VarianceDecayRate      = 1e-5;

    agent = rlDDPGAgent(actor, critic, agentOptions);

    % ---- Training options ----------------------------------------------
    trainingOptions                            = rlTrainingOptions;
    trainingOptions.MaxEpisodes                = getfielddef(ov, 'maxEpisodes', 4000);
    trainingOptions.MaxStepsPerEpisode         = params.simulation.Tf / params.simulation.Ts;
    trainingOptions.ScoreAveragingWindowLength = 100;
    trainingOptions.SaveAgentCriteria          = 'EpisodeReward';
    trainingOptions.SaveAgentValue             = getfielddef(ov, 'saveReward', 500);
    trainingOptions.Plots                      = 'training-progress';  % Episode Manager
    trainingOptions.Verbose                    = true;
    trainingOptions.StopOnError                = 'off';
    trainingOptions.StopTrainingCriteria       = 'AverageReward';
    trainingOptions.StopTrainingValue          = getfielddef(ov, 'stopReward', 1000);

    if useParallel
        trainingOptions.Parallelization = 'async';
        trainingOptions.ParallelizationOptions.StepsUntilDataIsSent = -1;
    end

    episodeReward = [];
    if doTrain
        trainingResults = train(agent, env, trainingOptions);
        episodeReward   = trainingResults.EpisodeReward(:);
        reset(agent);
        if useParallel; delete(gcp('nocreate')); end
    else
        if isempty(agentFile)
            s = load('sm_humanoid_walker_saved_agent');
            agent = s.saved_agent;
        else
            s = load(agentFile);
            [agent, episodeReward] = extractAgent(s);
        end
    end

    % ---- Validation rollout for the reward time-series ------------------
    open_system(mdlName);
    simOpts    = rlSimulationOptions('MaxSteps', 60 / agentOptions.SampleTime);
    experience = sim(env, agent, simOpts);
    rewardTime = experience.Reward.Time(:);
    rewardData = experience.Reward.Data(:);
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

function [agent, episodeReward] = extractAgent(s)
% Find an RL agent and (optionally) its training-reward curve inside a loaded
% .mat struct, by class rather than by variable name.
    agent = [];
    episodeReward = [];
    names = fieldnames(s);
    for i = 1:numel(names)
        val = s.(names{i});
        cls = class(val);
        if startsWith(cls, 'rl.agent.')
            agent = val;
        elseif isa(val, 'rl.train.result.rlTrainingResult') || isprop(val, 'EpisodeReward')
            try
                episodeReward = val.EpisodeReward(:);
            catch
            end
        end
    end
    if isempty(agent)
        error('gui_run_rl:noAgent', ...
              'No rl.agent.* object found in the selected .mat file.');
    end
end
