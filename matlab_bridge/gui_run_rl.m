function [episodeReward, rewardTime, rewardData] = gui_run_rl(overridesJson, doTrain, useParallel, agentFile, model)
% GUI_RUN_RL  Bridge called from the Python GUI to run an RL humanoid walker.
%
%   [episodeReward, rewardTime, rewardData] = ...
%       gui_run_rl(overridesJson, doTrain, useParallel, agentFile, model)
%
%   overridesJson : JSON string of parameter overrides (reward/controller
%                   fields, plus maxEpisodes / stopReward / saveReward).
%   doTrain       : logical. false => load a pretrained agent (see agentFile).
%   useParallel   : logical. Requires Parallel Computing Toolbox if true.
%   agentFile     : (optional) path to a .mat holding a trained agent. If given
%                   and doTrain is false, the agent is auto-detected by class.
%                   If empty, the shipped sm_humanoid_walker_saved_agent is used.
%   model         : (optional) 'ddpg' | 'td3' | 'sac'. Selects which agent to
%                   BUILD when training. Ignored when loading a pretrained agent
%                   (the loaded agent's own type is used). Default 'ddpg'.
%
%   The environment, reward, and Simulink model are identical across all three
%   algorithms; only the agent/networks differ. This lets you compare
%   algorithms on the same walking task.
%
%   NOTE: full training takes hours. Prefer doTrain=false for quick demos.

    if nargin < 5 || isempty(model);      model = 'ddpg';     end
    if nargin < 4;                        agentFile = '';     end
    if nargin < 3 || isempty(useParallel); useParallel = false; end
    if nargin < 2 || isempty(doTrain);    doTrain = false;    end
    model = lower(string(model));

    ov = struct();
    if ~isempty(overridesJson); ov = jsondecode(overridesJson); end

    % ---- Parameters -----------------------------------------------------
    sm_humanoid_walker_rl_parameters;    % defines `params`
    params = applyOverrides(params, ov); %#ok<NODEF>

    mdlName = 'sm_humanoid_walker_rl';
    load_system(mdlName);

    % ---- Environment (fixed spec, shared by every algorithm) ------------
    numAct          = 6;                 % 6 leg joints
    actionInfo      = rlNumericSpec([numAct 1], 'LowerLimit', -1, 'UpperLimit', 1);
    actionInfo.Name = 'jointDemands';

    numObs               = 25 + 6;       % 25 sensors + 6 previous actions
    observationInfo      = rlNumericSpec([numObs 1]);
    observationInfo.Name = 'observations';

    blk = [mdlName, '/RL Agent'];
    env = rlSimulinkEnv(mdlName, blk, observationInfo, actionInfo);

    Ts               = params.simulation.Ts;
    actorLayerSizes  = [400 300];
    criticLayerSizes = [400 300];

    % ---- Build the agent for the chosen algorithm -----------------------
    agent = [];
    if doTrain
        switch model
            case "ddpg"
                agent = buildDDPG(env, numObs, numAct, actorLayerSizes, ...
                                  criticLayerSizes, Ts);
            case "td3"
                agent = buildTD3(env, numObs, numAct, actorLayerSizes, ...
                                 criticLayerSizes, Ts);
            case "sac"
                agent = buildSAC(env, numObs, numAct, actorLayerSizes, ...
                                 criticLayerSizes, Ts);
            otherwise
                error('gui_run_rl:badModel', ...
                      'Unknown model "%s". Use ddpg, td3 or sac.', model);
        end
    end

    % ---- Training options (shared) --------------------------------------
    trainingOptions                            = rlTrainingOptions;
    trainingOptions.MaxEpisodes                = getfielddef(ov, 'maxEpisodes', 4000);
    trainingOptions.MaxStepsPerEpisode         = params.simulation.Tf / Ts;
    trainingOptions.ScoreAveragingWindowLength = 100;
    trainingOptions.SaveAgentCriteria          = 'EpisodeReward';
    trainingOptions.SaveAgentValue             = getfielddef(ov, 'saveReward', 500);
    trainingOptions.Plots                      = 'training-progress';
    trainingOptions.Verbose                    = true;
    trainingOptions.StopOnError                = 'off';
    trainingOptions.StopTrainingCriteria       = 'AverageReward';
    trainingOptions.StopTrainingValue          = getfielddef(ov, 'stopReward', 1000);

    if useParallel
        trainingOptions.Parallelization = 'async';
        trainingOptions.ParallelizationOptions.StepsUntilDataIsSent = -1;
    end

    % ---- Train or load --------------------------------------------------
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
    simOpts    = rlSimulationOptions('MaxSteps', 60 / Ts);
    experience = sim(env, agent, simOpts);
    rewardTime = experience.Reward.Time(:);
    rewardData = experience.Reward.Data(:);
end

% ======================================================================
%  Agent builders
% ======================================================================
function agent = buildDDPG(env, numObs, numAct, aSizes, cSizes, Ts)
    actor  = buildDeterministicActor(env, numObs, numAct, aSizes);
    critic = buildQCritic(env, numObs, numAct, cSizes);
    opt = rlDDPGAgentOptions;
    opt.SampleTime                          = Ts;
    opt.DiscountFactor                      = 0.99;
    opt.MiniBatchSize                       = 128;
    opt.ExperienceBufferLength              = 1e6;
    opt.TargetSmoothFactor                  = 1e-3;
    opt.NoiseOptions.MeanAttractionConstant = 5;
    opt.NoiseOptions.Variance               = 0.4;
    opt.NoiseOptions.VarianceDecayRate      = 1e-5;
    agent = rlDDPGAgent(actor, critic, opt);
end

function agent = buildTD3(env, numObs, numAct, aSizes, cSizes, Ts)
    % TD3 = twin critics + delayed policy updates. Reuses the deterministic
    % actor; builds two independently-initialised critics.
    actor   = buildDeterministicActor(env, numObs, numAct, aSizes);
    critic1 = buildQCritic(env, numObs, numAct, cSizes);
    critic2 = buildQCritic(env, numObs, numAct, cSizes);
    opt = rlTD3AgentOptions;
    opt.SampleTime                          = Ts;
    opt.DiscountFactor                      = 0.99;
    opt.MiniBatchSize                       = 128;
    opt.ExperienceBufferLength              = 1e6;
    opt.TargetSmoothFactor                  = 1e-3;
    opt.ExplorationModel.Variance           = 0.4;
    opt.ExplorationModel.VarianceDecayRate  = 1e-5;
    opt.TargetPolicySmoothModel.Variance    = 0.2;
    agent = rlTD3Agent(actor, [critic1 critic2], opt);
end

function agent = buildSAC(env, numObs, numAct, aSizes, cSizes, Ts)
    % SAC = stochastic (Gaussian) actor + twin critics + entropy tuning.
    actor   = buildGaussianActor(env, numObs, numAct, aSizes);
    critic1 = buildQCritic(env, numObs, numAct, cSizes);
    critic2 = buildQCritic(env, numObs, numAct, cSizes);
    opt = rlSACAgentOptions;
    opt.SampleTime             = Ts;
    opt.DiscountFactor         = 0.99;
    opt.MiniBatchSize          = 128;
    opt.ExperienceBufferLength = 1e6;
    opt.TargetSmoothFactor     = 1e-3;
    agent = rlSACAgent(actor, [critic1 critic2], opt);
end

% ======================================================================
%  Network builders
% ======================================================================
function actor = buildDeterministicActor(env, numObs, numAct, sizes)
    net = [
        featureInputLayer(numObs, 'Name', 'observation')
        fullyConnectedLayer(sizes(1), 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(sizes(2), 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(numAct, 'Name', 'fc3')
        tanhLayer('Name', 'tanh')
        ];
    net = dlnetwork(net);
    actor = rlContinuousDeterministicActor(net, ...
                env.getObservationInfo, env.getActionInfo);
end

function actor = buildGaussianActor(env, numObs, numAct, sizes)
    commonPath = [
        featureInputLayer(numObs, 'Name', 'observation')
        fullyConnectedLayer(sizes(1), 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(sizes(2), 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        ];
    meanPath = fullyConnectedLayer(numAct, 'Name', 'meanFC');
    stdPath  = [
        fullyConnectedLayer(numAct, 'Name', 'stdFC')
        softplusLayer('Name', 'std')     % keeps standard deviation positive
        ];
    net = layerGraph(commonPath);
    net = addLayers(net, meanPath);
    net = addLayers(net, stdPath);
    net = connectLayers(net, 'relu2', 'meanFC');
    net = connectLayers(net, 'relu2', 'stdFC');
    net = dlnetwork(net);
    actor = rlContinuousGaussianActor(net, ...
                env.getObservationInfo, env.getActionInfo, ...
                'ActionMeanOutputNames', 'meanFC', ...
                'ActionStandardDeviationOutputNames', 'std');
end

function critic = buildQCritic(env, numObs, numAct, sizes)
    statePath = [
        featureInputLayer(numObs, 'Name', 'observation')
        fullyConnectedLayer(sizes(1), 'Name', 'sfc1')
        reluLayer('Name', 'srelu1')
        fullyConnectedLayer(sizes(2), 'Name', 'sfc2')
        ];
    actionPath = [
        featureInputLayer(numAct, 'Name', 'action')
        fullyConnectedLayer(sizes(2), 'Name', 'afc1')
        ];
    commonPath = [
        additionLayer(2, 'Name', 'add')
        reluLayer('Name', 'crelu')
        fullyConnectedLayer(1, 'Name', 'qout')
        ];
    net = layerGraph(statePath);
    net = addLayers(net, actionPath);
    net = addLayers(net, commonPath);
    net = connectLayers(net, 'sfc2', 'add/in1');
    net = connectLayers(net, 'afc1', 'add/in2');
    net = dlnetwork(net);
    critic = rlQValueFunction(net, ...
                env.getObservationInfo, env.getActionInfo, ...
                'ObservationInputNames', 'observation', ...
                'ActionInputNames', 'action');
end

% ======================================================================
%  Helpers
% ======================================================================
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
        if startsWith(class(val), 'rl.agent.')
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
