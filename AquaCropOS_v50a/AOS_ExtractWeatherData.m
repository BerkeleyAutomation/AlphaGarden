function [Weather] = AOS_ExtractWeatherData(AOS_InitialiseStruct,...
    AOS_ClockStruct)
% Function to extract weather data for current time step

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

