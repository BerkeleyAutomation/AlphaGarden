function [GwStruct] = AOS_ReadGroundwaterTable(FileLocation,AOS_ClockStruct)
% Function to read input file and initialise groundwater table parameters

%% Read input file location %%
Location = FileLocation.Input;

%% Define empty structure %%
GwStruct = struct();

%% Read groundwater table input file %%
% Open file
filename = strcat(Location,FileLocation.GroundwaterFilename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining irrigation management
    % Throw error message
    fprintf(2,'Error - Groundwater input file not found\n');
end
WT = textscan(fileID,'%s',1,'delimiter',':','commentstyle','%%','headerlines',2);
WT = strtrim(WT);
Method = textscan(fileID,'%s',1,'delimiter',':','commentstyle','%%','headerlines',1);
Method = strtrim(Method);
if strcmp(WT{:},'N')
    % No water table present (don't read the rest of the input file)   
    GwStruct.WaterTable = 0;
    fclose(fileID);
elseif strcmp(WT{:},'Y')
    % Water table is present
    GwStruct.WaterTable = 1;
    GwStruct.Method = Method{:}{:};
    % Load and extract data  
    Data = textscan(fileID,'%f %f %f %f','delimiter','\t','commentstyle','%');
    fclose(fileID);
    GwDates = datenum(Data{1,3},Data{1,2},Data{1,1});
    GwDepths = Data{1,4}(:);
    nGwDates = length(GwDates);
    % Process and interpolate data
    StaDate = AOS_ClockStruct.SimulationStartDate;
    EndDate = AOS_ClockStruct.SimulationEndDate;
    Dates = StaDate:EndDate;
    nDates = length(Dates);
    if length(GwDepths) == 1
        % Only one value so water table elevation is constant for full
        % simulation period
        GwStruct.zGW = [Dates',ones(nDates,1)*GwDepths(1)];
    elseif length(GwDepths) > 1
        if strcmp(Method{:},'Constant')
            % No interpolation between dates
            staid = 1;
            GwLev = zeros(nDates,1);
            for ii = 1:nGwDates
                stoid = find(Dates==GwDates(ii),1,'first');
                GwLev(staid:stoid) = GwDepths(ii);
                staid = stoid+1;
            end
            if staid <= nDates
                GwLev(staid:nDates) = GwLev(staid-1);
            end
        elseif strcmp(Method{:},'Interp')
            % Linear interpolation between dates
            % Add start and end points (if they do not exist in input
            % values)
            if ismember(GwDates,StaDate) == false
                GwDates = [StaDate;GwDates];
                GwDepths = [GwDepths(min(GwDates));GwDepths];
            end
            if ismember(GwDates,EndDate) == false
                GwDates = [GwDates;EndDate];
                GwDepths = [GwDepths;GwDepths(max(GwDates))];
            end
            % Interpolate daily groundwater depths
            GwLev = interp1(GwDates',GwDepths',Dates');
        end
        % Assign values to output structure
        GwStruct.zGW = [Dates',GwLev];
    end
end

end