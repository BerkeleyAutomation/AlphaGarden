function [WeatherDB] = AOS_ReadWeatherInputs(FileLocation)
% Function to read and process input weather time-series

%% Declare global variables %%
global AOS_ClockStruct

%% Read input file location %%
Location = FileLocation.Input;

%% Read weather data inputs %%
% Open file
filename = FileLocation.WeatherFilename;
fileID = fopen(strcat(Location,filename));
if fileID == -1
    % Can't find text file defining weather inputs
    % Throw error message
    fprintf(2,'Error - Weather input file not found\n');
end

% Load data in
Data = textscan(fileID,'%f %f %f %f %f %f %f','delimiter','\t','headerlines',2);
fclose(fileID);

%% Convert dates to serial date format %%
Dates = datenum(Data{1,3},Data{1,2},Data{1,1});

%% Extract data %%
Tmin = Data{1,4};
Tmax = Data{1,5};
P = Data{1,6};
Et0 = Data{1,7};

%% Extract data for simulation period %%
% Find start and end dates
StartDate = AOS_ClockStruct.SimulationStartDate;
EndDate = AOS_ClockStruct.SimulationEndDate;
StartRow = find(Dates==StartDate);
EndRow = find(Dates==EndDate);

% Store data for simulation period
WeatherDB = [Dates(StartRow:EndRow),Tmin(StartRow:EndRow),...
    Tmax(StartRow:EndRow),P(StartRow:EndRow),...
    Et0(StartRow:EndRow)];

end