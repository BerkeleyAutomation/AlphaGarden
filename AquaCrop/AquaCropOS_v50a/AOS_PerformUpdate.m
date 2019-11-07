function [AOS_ClockStruct,AOS_InitialiseStruct,State] = AOS_PerformUpdate(AOS_ClockStruct,...
    AOS_InitialiseStruct,Irr)
% Function to run a single time-step (day) calculation of AquaCrop-OS

%% Get weather inputs for current time step %%
Weather = AOS_ExtractWeatherData(AOS_InitialiseStruct,AOS_ClockStruct);

%% Get model solution %%
[NewCond,Outputs,State] = AOS_Solution(AOS_ClockStruct,AOS_InitialiseStruct,...
    Weather,Irr);

%% Update initial conditions and outputs %%
AOS_InitialiseStruct.InitialCondition = NewCond;
AOS_InitialiseStruct.Outputs = Outputs;

%% Check model termination %%
[AOS_ClockStruct,AOS_InitialiseStruct] = AOS_CheckModelTermination(AOS_ClockStruct,...
    AOS_InitialiseStruct);

%% Update time step %%
[AOS_ClockStruct,AOS_InitialiseStruct] = AOS_UpdateTime(AOS_ClockStruct,...
    AOS_InitialiseStruct);

end