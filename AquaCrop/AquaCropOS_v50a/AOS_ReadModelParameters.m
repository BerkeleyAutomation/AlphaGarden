function [ParamStruct,CropChoices,FileLocation,AOS_ClockStruct] = AOS_ReadModelParameters(FileLocation,...
    AOS_ClockStruct)
% Function to read input files and initialise soil and crop parameters

%% Read input file location %%
Location = FileLocation.Input;

%% Read soil parameter input file %%
% Open file
filename = strcat(Location,FileLocation.SoilFilename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining soil parameters
    % Throw error message
    fprintf(2,'Error - Soil input file not found\n');
end

% Load data
SoilProfileName = textscan(fileID,'%s',1,'commentstyle','%%','headerlines',2);
SoilTextureName = textscan(fileID,'%s',1,'commentstyle','%%','headerlines',1);
SoilHydrologyName = textscan(fileID,'%s',1,'commentstyle','%%','headerlines',1);
DataArray = textscan(fileID,'%s %f','delimiter',':','commentstyle','%%');
fclose(fileID);

% Create assign string variables
FileLocation.SoilProfileFilename = SoilProfileName{:}{:};
FileLocation.SoilTextureFilename = SoilTextureName{:}{:};
FileLocation.SoilHydrologyFilename = SoilHydrologyName{:}{:};
ParamStruct.Soil = cell2struct(num2cell(DataArray{1,2}),strtrim(DataArray{1,1}));

%% Read soil profile input file %%
% Open file
filename = strcat(Location,FileLocation.SoilProfileFilename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining soil profile
    % Throw error message
    fprintf(2,'Error - Soil profile input file not found\n');
end

% Load data
Data = textscan(fileID,'%f %f %f','delimiter','\t','headerlines',2);
fclose(fileID);

% Create vector of soil compartments sizes and associated layers
ParamStruct.Soil.Comp.dz = Data{1,2}(:)';
ParamStruct.Soil.Comp.dzsum = round(100*(cumsum(ParamStruct.Soil.Comp.dz)))/100;
ParamStruct.Soil.Comp.Layer = Data{1,3}(:)';

%% Read crop mix input file %%
filename = strcat(Location,FileLocation.CropFilename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining crop mix parameters
    % Throw error message
    fprintf(2,'Error - Crop mix input file not found\n');
end
% Number of crops
nCrops = cell2mat(textscan(fileID,'%f',1,'headerlines',2));
% Crop rotation filename
Rotation = textscan(fileID,'%s',1,'headerlines',1,'commentstyle','%%');
% Crop rotation filename
RotationFilename = textscan(fileID,'%s',1,'headerlines',1,'commentstyle','%%');
% Crop information (type and filename)
CropInfo = textscan(fileID,'%s %s %s',nCrops,'delimiter',',',...
    'headerlines',2,'commentstyle','%%');
fclose(fileID);

%% Read crop parameter input files %%
% Create blank structure
ParamStruct.Crop = struct();
% Loop crop types
for ii = 1:nCrops
    % Open file
    filename = strcat(Location,CropInfo{1,2}{ii});
    fileID = fopen(filename);
    if fileID == -1
        % Can't find text file defining crop mix parameters
        % Throw error message
        fprintf(2,strcat('Error - Crop parameter input file, ',CropInfo{1,1}{ii},'not found\n'));
    end
    % Load data
    CropType = textscan(fileID,'%*s %f',1,'delimiter',':','commentstyle','%%','headerlines',2);
    CalendarType = textscan(fileID,'%*s %f',1,'delimiter',':','commentstyle','%%','headerlines',1);
    SwitchGDD = textscan(fileID,'%*s %f',1,'delimiter',':','commentstyle','%%','headerlines',1);
    PlantingDateStr = textscan(fileID,'%*s %s',1,'delimiter',':','commentstyle','%%','headerlines',1);
    HarvestDateStr = textscan(fileID,'%*s %s',1,'delimiter',':','commentstyle','%%','headerlines',1);
    DataArray = textscan(fileID,'%s %f','delimiter',':','commentstyle','%%');
    fclose(fileID);
    % Create crop parameter structure
    ParamStruct.Crop.(CropInfo{1,1}{ii}) = cell2struct(num2cell(DataArray{1,2}),strtrim(DataArray{1,1}));
    % Add additional parameters
    ParamStruct.Crop.(CropInfo{1,1}{ii}).CropType = cell2mat(CropType);
    ParamStruct.Crop.(CropInfo{1,1}{ii}).CalendarType = cell2mat(CalendarType);
    ParamStruct.Crop.(CropInfo{1,1}{ii}).SwitchGDD = cell2mat(SwitchGDD);
    ParamStruct.Crop.(CropInfo{1,1}{ii}).PlantingDate = PlantingDateStr{:}{:};
    ParamStruct.Crop.(CropInfo{1,1}{ii}).HarvestDate = HarvestDateStr{:}{:};
    % Add irrigation management information
    ParamStruct.Crop.(CropInfo{1,1}{ii}).IrrigationFile = CropInfo{1,3}{ii};
end

%% Find planting and harvest dates %%
if (nCrops > 1) || (strcmp(Rotation{1,1},'Y'))
    % Crop rotation occurs during the simulation period
    % Open rotation time-series file
    filename = strcat(Location,RotationFilename{1,1}{1});
    fileID = fopen(filename);
    if fileID == -1
        % Can't find text file defining crop rotation
        % Throw error message
        fprintf(2,'Error - Crop rotation input file not found\n');
    end
    % Load data
    DataArray = textscan(fileID,'%s %s %s','delimiter','\t','commentstyle',...
        '%%','headerlines',2);
    fclose(fileID);
    % Extract data
    PlantDates = datenum(DataArray{1,1},'dd/mm/yyyy');
    HarvestDates = datenum(DataArray{1,2},'dd/mm/yyyy');
    CropChoices = DataArray{1,3};
elseif nCrops == 1
    % Only one crop type considered during simulation - i.e. no rotations
    % either within or between yars
    % Get start and end years for full simulation
    SimStaDate = datevec(AOS_ClockStruct.SimulationStartDate);
    SimEndDate = datevec(AOS_ClockStruct.SimulationEndDate);
    % Get temporary crop structure
    CropTemp = ParamStruct.Crop.(CropInfo{1,1}{1});
    % Does growing season extend across multiple calendar years
    if datenum(CropTemp.PlantingDate,'dd/mm') < datenum(CropTemp.HarvestDate,'dd/mm')
        YrsPlant = SimStaDate(1):SimEndDate(1);
        YrsHarvest = YrsPlant;
    else
        YrsPlant = SimStaDate(1):SimEndDate(1)-1;
        YrsHarvest = SimStaDate(1)+1:SimEndDate(1);
    end
    % Correct for partial first growing season (may occur when simulating
    % off-season soil water balance)
    if datenum(strcat(CropTemp.PlantingDate,'/',num2str(YrsPlant(1))),...
            'dd/mm/yyyy') < AOS_ClockStruct.SimulationStartDate
        YrsPlant = YrsPlant(2:end);
        YrsHarvest = YrsHarvest(2:end);
    end
    % Define blank variables
    PlantDates = zeros(length(YrsPlant),1);
    HarvestDates = zeros(length(YrsHarvest),1);
    CropChoices = cell(length(YrsPlant),1);
    % Determine planting and harvest dates
    for ii = 1:length(YrsPlant)
        PlantDates(ii) = datenum(strcat(CropTemp.PlantingDate,'/',...
            num2str(YrsPlant(ii))),'dd/mm/yyyy');
        HarvestDates(ii) = datenum(strcat(CropTemp.HarvestDate,'/',...
            num2str(YrsHarvest(ii))),'dd/mm/yyyy');
        CropChoices{ii} = CropInfo{1,1}{1};
    end
end

%% Update clock parameters %%
% Store planting and harvest dates
AOS_ClockStruct.PlantingDate = PlantDates;
AOS_ClockStruct.HarvestDate = HarvestDates;
AOS_ClockStruct.nSeasons = length(PlantDates);
% Initialise growing season counter
if AOS_ClockStruct.StepStartTime == AOS_ClockStruct.PlantingDate(1) 
    AOS_ClockStruct.SeasonCounter = 1;
else
    AOS_ClockStruct.SeasonCounter = 0;
end

end