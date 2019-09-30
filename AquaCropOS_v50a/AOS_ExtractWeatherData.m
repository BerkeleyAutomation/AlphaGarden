function [Weather] = AOS_ExtractWeatherData()
% Function to extract weather data for current time step

%% Define global variables %%
global AOS_ClockStruct
global AOS_InitialiseStruct

%% Extract weather dataset %%
WeatherDB = AOS_InitialiseStruct.Weather;

%% Extract weather data for current time step %%
% Get current date
Date = AOS_ClockStruct.StepStartTime;
% Find row corresponding to the current date in dataset
Row = WeatherDB(:,1)==Date;
% Get weather variables
Weather.MinTemp = WeatherDB(Row,2);
Weather.MaxTemp = WeatherDB(Row,3);
Weather.Precipitation = WeatherDB(Row,4);
Weather.ReferenceET = WeatherDB(Row,5);

end

