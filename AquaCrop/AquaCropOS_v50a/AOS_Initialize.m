function [AOS_ClockStruct,AOS_InitialiseStruct] = AOS_Initialize()
% Function to initialise AquaCrop-OS

%% Get file locations %%
FileLocation = AOS_ReadFileLocations();

%% Define model run time %%
AOS_ClockStruct = AOS_ReadClockParameters(FileLocation);

%% Read climate data %%
[WeatherStruct, AOS_ClockStruct] = AOS_ReadWeatherInputs(FileLocation,AOS_ClockStruct);

%% Read model parameter files %%
[ParamStruct,CropChoices,FileLocation,AOS_ClockStruct] = AOS_ReadModelParameters(FileLocation,...
    AOS_ClockStruct);

%% Read field management file %%
FieldMngtStruct = AOS_ReadFieldManagement(FileLocation);

%% Read groundwater table file %%
GwStruct = AOS_ReadGroundwaterTable(FileLocation,AOS_ClockStruct);

%% Compute additional variables %%
ParamStruct = AOS_ComputeVariables(ParamStruct,WeatherStruct,...
    AOS_ClockStruct,GwStruct,CropChoices,FileLocation,AOS_ClockStruct);

%% Define initial conditions %%
InitCondStruct = AOS_ReadModelInitialConditions(ParamStruct,GwStruct,...
    FieldMngtStruct,CropChoices,FileLocation,AOS_ClockStruct);

%% Pack output structure %%
AOS_InitialiseStruct = struct();
AOS_InitialiseStruct.Parameter = ParamStruct;
AOS_InitialiseStruct.FieldManagement = FieldMngtStruct;
AOS_InitialiseStruct.Groundwater = GwStruct;
AOS_InitialiseStruct.InitialCondition = InitCondStruct;
AOS_InitialiseStruct.CropChoices = CropChoices;
AOS_InitialiseStruct.Weather = WeatherStruct;
AOS_InitialiseStruct.FileLocation = FileLocation;

%% Setup output files %%
% Define output file location
FileLoc = FileLocation.Output;
% Setup blank matrices to store outputs
AOS_InitialiseStruct.Outputs.WaterContents = zeros(...
    length(AOS_ClockStruct.TimeSpan),5+ParamStruct.Soil.nComp);
AOS_InitialiseStruct.Outputs.WaterFluxes = zeros(...
    length(AOS_ClockStruct.TimeSpan),18);
AOS_InitialiseStruct.Outputs.CropGrowth = zeros(...
    length(AOS_ClockStruct.TimeSpan),15);
AOS_InitialiseStruct.Outputs.FinalOutput = cell(AOS_ClockStruct.nSeasons,8);
% Store dates in daily matrices
Dates = datevec(AOS_ClockStruct.TimeSpan);
AOS_InitialiseStruct.Outputs.WaterContents(:,1:3) = Dates(:,1:3);
AOS_InitialiseStruct.Outputs.WaterFluxes(:,1:3) = Dates(:,1:3);
AOS_InitialiseStruct.Outputs.CropGrowth(:,1:3) = Dates(:,1:3);

if strcmp(AOS_InitialiseStruct.FileLocation.WriteDaily,'Y')
    % Water contents (daily)
    names = cell(1,ParamStruct.Soil.nComp);
    for ii = 1:ParamStruct.Soil.nComp
        z = ParamStruct.Soil.Comp.dzsum(ii)-(ParamStruct.Soil.Comp.dz(ii)/2);
        names{ii} = strcat(num2str(z),'m');
    end
    fid = fopen(strcat(FileLoc,FileLocation.OutputFilename,'_WaterContents.txt'),'w+t');
    fprintf(fid,strcat(repmat('%s\t',1,ParamStruct.Soil.nComp+5),'\n'),...
        'Year','Month','Day','SimDay','Season',names{:});
    fclose(fid);

    % Hydrological fluxes (daily)
    fid = fopen(strcat(FileLoc,FileLocation.OutputFilename,'_WaterFluxes.txt'),'w+t');
    fprintf(fid,strcat(repmat('%s\t',1,18),'\n'),'Year','Month','Day',...
        'SimDay','Season','wRZ','zGW','wSurf','Irr','Infl','RO','DP','CR',...
        'GWin','Es','EsX','Tr','TrX');
    fclose(fid);

    % Crop growth (daily)
    fid = fopen(strcat(FileLoc,FileLocation.OutputFilename,'_CropGrowth.txt'),'w+t');
    fprintf(fid,strcat(repmat('%s\t',1,15),'\n'),'Year','Month','Day',...
        'SimDay','Season','GDD','TotGDD','Root Depth','CC','RefCC','Bio',...
        'RefBio','HI','HIadj','Yield');
    fclose(fid);
end

% Final output (at end of each growing season)
fid = fopen(strcat(FileLoc,FileLocation.OutputFilename,'_FinalOutput.txt'),'w+t');
fprintf(fid,strcat(repmat('%s\t',1,8),'\n'),'GSeason','Crop','PlantD',...
    'PlantSD','HarvestCD','HarvestSD','Yield','TotIrr');
fclose(fid);

%save(strcat(FileLocation.Output,'AOS_InitialiseStruct'),'AOS_InitialiseStruct');
%save(strcat(FileLocation.Output,'AOS_ClockStruct'),'AOS_ClockStruct');
end

