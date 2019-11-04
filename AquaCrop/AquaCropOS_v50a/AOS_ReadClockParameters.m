function [ClockStruct] = AOS_ReadClockParameters(FileLocation)
% Function to read input files and initialise model clock parameters

%% Read input file location %%
Location = FileLocation.Input;

%% Read clock parameter input file %%
filename = strcat(Location,FileLocation.ClockFilename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining clock parameters
    % Throw error message
    fprintf(2,'Error - Clock input file not found\n');
end
% Load data
DataArray = textscan(fileID,'%s %s','delimiter',':','commentstyle','%');
fclose(fileID);
% Create and assign variables
ClockStruct = cell2struct(DataArray{1,2},strtrim(DataArray{1,1}));

%% Define clock parameters %%
% Initialise time step counter
ClockStruct.TimeStepCounter = 1;
% Initialise model termination condition
ClockStruct.ModelTermination = false;
% Simulation start time as serial date number
DateStaV = datevec(ClockStruct.SimulationStartTime,'yyyy-mm-dd');
ClockStruct.SimulationStartDate = datenum(DateStaV);
% Simulation end time as serial date number
DateStoV = datevec(ClockStruct.SimulationEndTime,'yyyy-mm-dd');
ClockStruct.SimulationEndDate = datenum(DateStoV);
% Time step (years)
ClockStruct.TimeStep = 1;
% Total numbers of time steps (days)
ClockStruct.nSteps = ClockStruct.SimulationEndDate-...
    ClockStruct.SimulationStartDate;
% Time spans
TimeSpan = zeros(1,ClockStruct.nSteps+1);
TimeSpan(1) = ClockStruct.SimulationStartDate;
TimeSpan(end) = ClockStruct.SimulationEndDate;
for ss = 2:ClockStruct.nSteps
    TimeSpan(ss) = TimeSpan(ss-1)+1;
end
ClockStruct.TimeSpan = TimeSpan;
% Time at start of current time step
ClockStruct.StepStartTime = ClockStruct.TimeSpan(ClockStruct.TimeStepCounter);
% Time at end of current time step 
ClockStruct.StepEndTime = ClockStruct.TimeSpan(ClockStruct.TimeStepCounter+1);
% Number of time-steps (per day) for soil evaporation calculation
ClockStruct.EvapTimeSteps = 20;

end