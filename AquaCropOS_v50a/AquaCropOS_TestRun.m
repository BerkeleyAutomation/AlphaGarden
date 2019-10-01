function [] = AquaCropOS_TestRun()

%% Run model %%
% Initialise simulation
[AOS_ClockStruct,AOS_InitialiseStruct] = AOS_Initialize();

% Perform single time-step (day)
while AOS_ClockStruct.ModelTermination == false
   [AOS_ClockStruct,AOS_InitialiseStruct,State] = AOS_PerformUpdate(AOS_ClockStruct,...
       AOS_InitialiseStruct,10); % Specify irrigation amount here
end

% Finish simulation
AOS_Finish(AOS_ClockStruct,AOS_InitialiseStruct);

end