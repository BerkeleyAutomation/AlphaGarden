function [IrrMngtStruct,FileLocation] = AOS_ReadIrrigationManagement...
    (ParamStruct,FileLocation)
% Function to read and initialise irrigation management parameters

%% Declare global variables %%
global AOS_ClockStruct

%% Read AOS input file location %%
Location = FileLocation.Input;

%% Read irrigation management input files %%
% Check for number of crop types
Crops = fieldnames(ParamStruct.Crop);
nCrops = length(Crops);
% Create blank structure
IrrMngtStruct = struct();

for ii = 1:nCrops
    % Open file
    filename = strcat(Location,ParamStruct.Crop.(Crops{ii}).IrrigationFile);
    fileID = fopen(filename);
    if fileID == -1
        % Can't find text file defining irrigation management
        % Throw error message
        fprintf(2,strcat('Error - Irrigation management input file not found for ',Crops{ii},'\n'));
    end
    % Load data
    IrrSchFilename = textscan(fileID,'%s',1,'commentstyle','%%','headerlines',2);
    DataArray = textscan(fileID,'%s %f','delimiter',':','commentstyle','%%');
    fclose(fileID);
    
    % Create and assign numeric variables
    IrrMngtStruct.(Crops{ii}) = cell2struct(num2cell(DataArray{1,2}),strtrim(DataArray{1,1}));
    
    % Consolidate soil moisture targets in to one variable
    IrrMngtStruct.(Crops{ii}).SMT = [IrrMngtStruct.(Crops{ii}).SMT1,...
        IrrMngtStruct.(Crops{ii}).SMT2,IrrMngtStruct.(Crops{ii}).SMT3,...
        IrrMngtStruct.(Crops{ii}).SMT4];
    IrrMngtStruct.(Crops{ii}) = rmfield(IrrMngtStruct.(Crops{ii}),...
        {'SMT1','SMT2','SMT3','SMT4'});
    
    % If specified, read input irrigation time-series
    if IrrMngtStruct.(Crops{ii}).IrrMethod == 3
        % Load data
        filename = strcat(Location,IrrSchFilename{:}{:});
        fileID = fopen(filename);
        if fileID == -1
            % Can't find text file defining irrigation schedule
            % Throw error message
            fprintf(2,strcat('Error - Irrigation schedule input file not found for ',Crops{ii},'\n'));
        end
        Data = textscan(fileID,'%f %f %f %f','delimiter','\t','headerlines',2);
        fclose(fileID);
        
        % Extract data
        IrrEvents = Data{1,4}(:);
        % Convert dates to serial date format
        IrrDates = datenum(datenum(Data{1,3},Data{1,2},Data{1,1}));
        % Create full time series
        StartDate = AOS_ClockStruct.SimulationStartDate;
        EndDate = AOS_ClockStruct.SimulationEndDate;
        Dates = StartDate:EndDate;
        Irr = zeros(length(Dates),1);
        idx = ismember(Dates,IrrDates);
        Irr(idx) = IrrEvents;
        IrrigationSch = [Dates',Irr];
        IrrMngtStruct.(Crops{ii}).IrrigationSch = IrrigationSch;
    end
end

end