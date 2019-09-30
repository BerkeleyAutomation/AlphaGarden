function [InitCondStruct] = AOS_ReadModelInitialConditions(ParamStruct,...
    GwStruct,FieldMngtStruct,CropChoices,FileLocation)
% Function to set up initial model conditions

%% Declare global variables %%
global AOS_ClockStruct

%% Define initial conditions %%
InitCondStruct = struct();

% Counters
[InitCondStruct.AgeDays,InitCondStruct.AgeDays_NS,...
    InitCondStruct.AerDays,InitCondStruct.IrrCum,InitCondStruct.DelayedGDDs,...
    InitCondStruct.DelayedCDs,InitCondStruct.PctLagPhase,...
    InitCondStruct.tEarlySen,InitCondStruct.GDDcum,...
    InitCondStruct.DaySubmerged,InitCondStruct.IrrNetCum,...
    InitCondStruct.DAP,InitCondStruct.Epot,InitCondStruct.Tpot] = deal(0);

% States
[InitCondStruct.PreAdj,InitCondStruct.CropMature,InitCondStruct.CropDead,...
    InitCondStruct.Germination,InitCondStruct.PrematSenes,...
    InitCondStruct.HarvestFlag] = deal(false);

% Harvest index
[InitCondStruct.Stage,InitCondStruct.Fpre,InitCondStruct.Fpost,...
    InitCondStruct.fpost_dwn,InitCondStruct.fpost_upp] = deal(1);
[InitCondStruct.HIcor_Asum,InitCondStruct.HIcor_Bsum,...
    InitCondStruct.Fpol,InitCondStruct.sCor1,InitCondStruct.sCor2] = deal(0);

% Growth stage
InitCondStruct.GrowthStage = 0;

% Aeration stress (compartment level)
InitCondStruct.AerDaysComp = zeros(1,ParamStruct.Soil.nComp);

% Transpiration
InitCondStruct.TrRatio = 1;

% Crop growth
[InitCondStruct.CC,InitCondStruct.CCadj,InitCondStruct.CC_NS,...
    InitCondStruct.CCadj_NS,InitCondStruct.Zroot,InitCondStruct.B,...
    InitCondStruct.B_NS,InitCondStruct.HI,InitCondStruct.HIadj,...
    InitCondStruct.CCxAct,InitCondStruct.CCxAct_NS,...
    InitCondStruct.CCxW,InitCondStruct.CCxW_NS,...
    InitCondStruct.CCxEarlySen,InitCondStruct.CCprev] = deal(0);
InitCondStruct.rCor = 1;
if AOS_ClockStruct.SeasonCounter == 0
    InitCondStruct.Zroot = 0;
    InitCondStruct.CC0adj = 0;
elseif AOS_ClockStruct.SeasonCounter == 1
    InitCondStruct.Zroot = ParamStruct.Crop.(CropChoices...
        {AOS_ClockStruct.SeasonCounter}).Zmin;
    InitCondStruct.CC0adj = ParamStruct.Crop.(CropChoices...
        {AOS_ClockStruct.SeasonCounter}).CC0;
end
    
% Surface storage between bunds
if (FieldMngtStruct.Bunds == 0) && (FieldMngtStruct.zBund > 0.001)
    % Get initial storage between surface bunds
    InitCondStruct.SurfaceStorage = FieldMngtStruct.BundWater;
    if InitCondStruct.SurfaceStorage > FieldMngtStruct.zBund
        InitCondStruct.SurfaceStorage = FieldMngtStruct.zBund;
    end
    InitCondStruct.SurfaceStorageIni = InitCondStruct.SurfaceStorage;
else
    % No surface bunds
    InitCondStruct.SurfaceStorage = 0;
    InitCondStruct.SurfaceStorageIni = 0;
end

%% Check for presence of groundwater table %%
if GwStruct.WaterTable == 0 % No water table present
    % Set initial groundwater level to dummy value
    InitCondStruct.zGW = -999;
    InitCondStruct.WTinSoil = false;
    % Set adjusted field capacity to default field capacity
    InitCondStruct.th_fc_Adj = ParamStruct.Soil.Comp.th_fc;
elseif GwStruct.WaterTable == 1 % Water table is present
    % Set initial groundwater level
    InitCondStruct.zGW = GwStruct.zGW((GwStruct.zGW(:,1)==AOS_ClockStruct.StepStartTime),2);
    % Find compartment mid-points
    zBot = cumsum(ParamStruct.Soil.Comp.dz);
    zTop = zBot-ParamStruct.Soil.Comp.dz;
    zMid = (zTop+zBot)/2;
    % Check if water table is within modelled soil profile %%
    if InitCondStruct.zGW >= 0
        idx = find(zMid>=InitCondStruct.zGW,1);
        if isempty(idx)
            InitCondStruct.WTinSoil = false;
        else
            InitCondStruct.WTinSoil = true;
        end
    else
        InitCondStruct.WTinSoil = false;
    end
    % Adjust compartment field capacity
    compi = ParamStruct.Soil.nComp;
    thfcAdj = zeros(1,compi);
    while compi >= 1
        layeri = ParamStruct.Soil.Comp.Layer(compi);
        if ParamStruct.Soil.Layer.th_fc(layeri) <= 0.1
            Xmax = 1;
        else
            if ParamStruct.Soil.Layer.th_fc(layeri) >= 0.3
                Xmax = 2;
            else
                pF = 2+0.3*(ParamStruct.Soil.Layer.th_fc(layeri)-0.1)/0.2;
                Xmax = (exp(pF*log(10)))/100;
            end
        end
        if (InitCondStruct.zGW < 0) || ((InitCondStruct.zGW-zMid(compi)) >= Xmax)
            for ii = 1:compi
                layerii = ParamStruct.Soil.Comp.Layer(ii);
                thfcAdj(ii) = ParamStruct.Soil.Layer.th_fc(layerii);
            end
            compi = 0;
        else
            if ParamStruct.Soil.Layer.th_fc(layeri) >= ParamStruct.Soil.Layer.th_s(layeri)
                thfcAdj(compi) = ParamStruct.Soil.Layer.th_fc(layeri);
            else
                if zMid(compi) >= InitCondStruct.zGW
                    thfcAdj(compi) = ParamStruct.Soil.Layer.th_s(layeri);
                else
                    dV = ParamStruct.Soil.Layer.th_s(layeri)-...
                        ParamStruct.Soil.Layer.th_fc(layeri);
                    dFC = (dV/(Xmax^2))*((zMid(compi)-(InitCondStruct.zGW-Xmax))^2);
                    thfcAdj(compi) = ParamStruct.Soil.Layer.th_fc(layeri)+dFC;
                end
            end
            compi = compi-1;
        end
    end
    % Store adjusted field capacity values
    InitCondStruct.th_fc_Adj = round((thfcAdj*1000))/1000;
end

%% Define initial water contents %%
% Read input file
Location = FileLocation.Input;
filename = strcat(Location,FileLocation.InitialWCFilename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining soil initial water content
    % Throw error message
    fprintf(2,'Error - Initial water content input file not found\n');
end
TypeStr = textscan(fileID,'%s',1,'delimiter',':','commentstyle','%%','headerlines',2);
MethodStr = textscan(fileID,'%s',1,'delimiter',':','commentstyle','%%','headerlines',1);
NbrPts = textscan(fileID,'%d',1,'delimiter',':','commentstyle','%%','headerlines',1);
NbrPts = cell2mat(NbrPts);
Data_Pts = textscan(fileID,'%f %s',NbrPts,'delimiter','\t','commentstyle','%%',...
'headerlines',1);
fclose(fileID);

% Extract data
Locs = cell2mat(Data_Pts(:,1));
Vals = zeros(length(Locs),1);
Method = MethodStr{:}{:};
Type = TypeStr{:}{:};
if strcmp(Method,'Depth')
    Locs = round((100*Locs))/100;
end

% Define soil compartment depths and layers
SoilLayers = ParamStruct.Soil.Comp.Layer;
SoilDepths = cumsum(ParamStruct.Soil.Comp.dz);
SoilDepths = round((100*SoilDepths))/100;

% Assign data
if strcmp(Type,'Num')
    % Values are defined as numbers (m3/m3) so no calculation required
    Vals = cellfun(@str2num,Data_Pts{1,2});
elseif strcmp(Type,'Pct')
    % Values are defined as percentage of TAW. Extract and assign value for
    % each soil layer based on calculated/input soil hydraulic properties
    ValsTmp = cellfun(@str2num,Data_Pts{1,2});
    for ii = 1:length(ValsTmp)
        if strcmp(Method,'Depth')
            % Find layer at specified depth
            if Locs(ii) < SoilDepths(end)
                LayTmp = SoilLayers(find(Locs(ii)>=SoilDepths,1,'first')); 
            else
                LayTmp = SoilLayers(end);
            end
            % Calculate moisture content at specified depth
            Vals(ii) = ParamStruct.Soil.Layer.th_wp(LayTmp)+...
                ((ValsTmp(ii)/100)*(ParamStruct.Soil.Layer.th_fc(LayTmp)-...
                ParamStruct.Soil.Layer.th_wp(LayTmp)));
        elseif strcmp(Method,'Layer')
            % Calculate moisture content at specified layer
            LayTmp = Locs(ii);
            Vals(ii) = ParamStruct.Soil.Layer.th_wp(LayTmp)+...
                ((ValsTmp(ii)/100)*(ParamStruct.Soil.Layer.th_fc(LayTmp)-...
                ParamStruct.Soil.Layer.th_wp(LayTmp)));
        end
    end
elseif strcmp(Type,'Prop')
    % Values are specified as soil hydraulic properties (SAT, FC, or WP).
    % Extract and assign value for each soil layer
    ValsTmp = Data_Pts(:,2);
    for ii = 1:size(ValsTmp{:},1)
        if strcmp(Method,'Depth')
            % Find layer at specified depth
            if Locs(ii) < SoilDepths(end)
                LayTmp = SoilLayers(find(Locs(ii)>=SoilDepths,1,'first')); 
            else
                LayTmp = SoilLayers(end);
            end
            % Calculate moisture content at specified depth
            if strcmp(ValsTmp{1,1}(ii),'SAT')
                Vals(ii) = ParamStruct.Soil.Layer.th_s(LayTmp);
            elseif strcmp(ValsTmp{1,1}(ii),'FC')
                Vals(ii) = ParamStruct.Soil.Layer.th_fc(LayTmp);
            elseif strcmp(ValsTmp{1,1}(ii),'WP')
                Vals(ii) = ParamStruct.Soil.Layer.th_wp(LayTmp);
            end
        elseif strcmp(Method,'Layer')
            % Calculate moisture content at specified layer
            LayTmp = Locs(ii);
            if strcmp(ValsTmp{1,1}(ii),'SAT')
                Vals(ii) = ParamStruct.Soil.Layer.th_s(LayTmp);
            elseif strcmp(ValsTmp{1,1}(ii),'FC')
                Vals(ii) = ParamStruct.Soil.Layer.th_fc(LayTmp);
            elseif strcmp(ValsTmp{1,1}(ii),'WP')
                Vals(ii) = ParamStruct.Soil.Layer.th_wp(LayTmp);
            end
        end 
    end
end

% Interpolate values to all soil compartments
thini = zeros(1,ParamStruct.Soil.nComp);
if strcmp(Method,'Layer')
    for ii = 1:length(Vals)
        thini(ParamStruct.Soil.Comp.Layer==Locs(ii)) = Vals(ii);
    end
    InitCondStruct.th = thini;
elseif strcmp(Method,'Depth')
    % Add zero point
    if Locs(1) > 0
        Locs = [0;Locs];
        Vals = [Vals(1);Vals];
    end
    % Add end point (bottom of soil profile)
    if Locs(end) < ParamStruct.Soil.zSoil
        Locs = [Locs;ParamStruct.Soil.zSoil];
        Vals = [Vals;Vals(end)];
    end
    % Find centroids of compartments
    comp_top = [0,SoilDepths(1:end-1)];
    comp_bot = SoilDepths(1:end);
    comp_mid = (comp_top+comp_bot)/2;
    % Interpolate initial water contents to each compartment
    thini = interp1(Locs',Vals',comp_mid);
    InitCondStruct.th = thini;
end

% If groundwater table is present and calculating water contents based on
% field capacity, then reset value to account for possible changes in field
% capacity caused by capillary rise effects
if GwStruct.WaterTable == 1
    if (strcmp(Type,'Prop')) && (strcmp(ValsTmp{1,1}(ii),'FC'))
        InitCondStruct.th = InitCondStruct.th_fc_Adj; 
    end
end

% If groundwater table is present in soil profile then set all water
% contents below the water table to saturation
if InitCondStruct.WTinSoil == true
    % Find compartment mid-points
    zBot = cumsum(ParamStruct.Soil.Comp.dz);
    zTop = zBot-ParamStruct.Soil.Comp.dz;
    zMid = (zTop+zBot)/2;
    idx = find(zMid >= InitCondStruct.zGW,1);
    for ii = idx:ParamStruct.Soil.nComp
        layeri = ParamStruct.Soil.Comp.Layer(ii);
        InitCondStruct.th(ii) = ParamStruct.Soil.Layer.th_s(layeri);
    end
end

InitCondStruct.thini = InitCondStruct.th;

end