function [] = AOS_PerformTimeStep()
% Function to run a single time-step (day) calculation of AquaCrop-OS

%% Define global variables %%
global AOS_InitialiseStruct

%% Get weather inputs for current time step %%
Weather = AOS_ExtractWeatherData();

%% Get model solution %%
[NewCond,Outputs] = AOS_Solution(Weather);

%% Update initial conditions and outputs %%
AOS_InitialiseStruct.InitialCondition = NewCond;
AOS_InitialiseStruct.Outputs = Outputs;

%% Check model termination %%
AOS_CheckModelTermination();

%% Update time step %%
AOS_UpdateTime();

end