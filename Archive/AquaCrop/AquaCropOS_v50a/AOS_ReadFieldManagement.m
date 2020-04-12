function [FieldMngtStruct] = AOS_ReadFieldManagement(FileLocation)
% Function to read input files and initialise field management parameters

%% Get input file location %%
Location = FileLocation.Input;

%% Read field management parameter input files %%
% Open file
filename = strcat(Location,FileLocation.FieldManagementFilename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining irrigation management
    % Throw error message
    fprintf(2,'Error - Field management input file not found\n');
end
% Load data
DataArray = textscan(fileID,'%s %f','delimiter',':','commentstyle','%%');
fclose(fileID);

% Create and assign variables
FieldMngtStruct = cell2struct(num2cell(DataArray{1,2}),strtrim(DataArray{1,1}));

end