function [] = AquaCropOS_BatchRUN(FileLocIn,FileLocOut)
% ---------------------------------------------------------------------- %
% Tim Foster                                                             %
% June 2016                                                              %
%                                                                        %
% Function to perform batch run of AquaCrop-OS v5.0a                     %
%                                                                        %
% ---------------------------------------------------------------------- %

%% Declare global variables %%
global AOS_ClockStruct
global AOS_FileLoc

%% Store batch run number in global variable %%
AOS_FileLoc = struct();
AOS_FileLoc.Input = FileLocIn;
AOS_FileLoc.Output = FileLocOut;

%% Run model %%
% Initialise
AOS_Initialize();

% Perform Time Step
while AOS_ClockStruct.ModelTermination == false
   AOS_PerformTimeStep();
end

% Finish 
AOS_Finish();

end