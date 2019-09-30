function [FileLocation] = AOS_ReadFileLocations()
% Function to read input and output file locations

%% Declare global variables %%
global AOS_FileLoc

%% Read AOS file location input file %%
filename = 'FileLocations.txt';
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining locations of input and output folders.
    % Throw error message
    fprintf(2,'Error - File location input file not found\n');
end

%% Read file locations %%
% Load data
DataArray = textscan(fileID,'%s','delimiter','\n','commentstyle','%%');
fclose(fileID);

% Store input and output folder strings
if isempty(AOS_FileLoc) ==  1
    % Use default input and output folder locations (not in batch mode)
    InputStr = regexp(DataArray{1}{1},'/','split');
    OutputStr = regexp(DataArray{1}{2},'/','split');
    FileLocation = struct();
    if strcmp(filesep,'/') == true
        FileLocation.Input = strcat(filesep,fullfile(InputStr{:}),filesep);
        FileLocation.Output = strcat(filesep,fullfile(OutputStr{:}),filesep);
    else
        FileLocation.Input = strcat(fullfile(InputStr{:}),filesep);
        FileLocation.Output = strcat(fullfile(OutputStr{:}),filesep);
    end
else
    % Define input and output folders (in batch mode)
    InputStr = regexp(AOS_FileLoc.Input{:},'/','split');
    OutputStr = regexp(AOS_FileLoc.Output{:},'/','split');
    FileLocation = struct();
    if strcmp(filesep,'/') == true
        FileLocation.Input = strcat(filesep,fullfile(InputStr{:}),filesep);
        FileLocation.Output = strcat(filesep,fullfile(OutputStr{:}),filesep);
    else
        FileLocation.Input = strcat(fullfile(InputStr{:}),filesep);
        FileLocation.Output = strcat(fullfile(OutputStr{:}),filesep);
    end
end
    
%% Read file setup %%
% Load data
filename = 'FileSetup.txt';
fileID = fopen(strcat(FileLocation.Input,filename));
if fileID == -1
    % Can't find text file defining file setup.
    % Throw error message
    fprintf(2,'Error - File setup input file not found\n');
end
DataArray = textscan(fileID,'%s','delimiter','\n','commentstyle','%%');
fclose(fileID);

% Store strings
FileLocation.ClockFilename = DataArray{1}{1};
FileLocation.WeatherFilename = DataArray{1}{2};
FileLocation.CO2Filename = DataArray{1}{3};
FileLocation.CropFilename = DataArray{1}{4};
FileLocation.SoilFilename = DataArray{1}{5};
FileLocation.FieldManagementFilename = DataArray{1}{6};
FileLocation.InitialWCFilename = DataArray{1}{7};
FileLocation.GroundwaterFilename = DataArray{1}{8};
FileLocation.OutputFilename = DataArray{1}{9};
FileLocation.WriteDaily = DataArray{1}{10};

end