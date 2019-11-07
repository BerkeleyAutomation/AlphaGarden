function [AOS_InitialiseStruct] = AOS_ResetInitialConditions(AOS_InitialiseStruct,...
    AOS_ClockStruct)
% Function to reset initial model conditions for start of growing
% season (when running model over multiple seasons)

%% Extract crop type %%
CropType = AOS_InitialiseStruct.CropChoices{AOS_ClockStruct.SeasonCounter};

%% Extract structures for updating %%
InitCond = AOS_InitialiseStruct.InitialCondition;
Soil = AOS_InitialiseStruct.Parameter.Soil;
Crop = AOS_InitialiseStruct.Parameter.Crop.(CropType);
CO2 = AOS_InitialiseStruct.Parameter.CO2;

%% Reset counters %%
[InitCond.AgeDays,InitCond.AgeDays_NS,...
    InitCond.AerDays,InitCond.IrrCum,...
    InitCond.DelayedGDDs,InitCond.DelayedCDs,...
    InitCond.tEarlySen,InitCond.GDDcum,...
    InitCond.DaySubmerged,InitCond.IrrNetCum,...
    InitCond.DAP,InitCond.PctLagPhase] = deal(0);

InitCond.AerDaysComp = zeros(1,Soil.nComp);

%% Reset states %%
% General states
[InitCond.PreAdj,InitCond.CropMature,InitCond.CropDead,...
    InitCond.Germination,InitCond.PrematSenes,...
    InitCond.HarvestFlag] = deal(false);

% Harvest index
[InitCond.Stage,InitCond.Fpre,InitCond.Fpost,...
    InitCond.fpost_dwn,InitCond.fpost_upp] = deal(1);
[InitCond.HIcor_Asum,InitCond.HIcor_Bsum,...
    InitCond.Fpol,InitCond.sCor1,InitCond.sCor2] = deal(0);

% Growth stage
InitCond.GrowthStage = 0;

% Transpiration
InitCond.TrRatio = 1;

% Crop growth
[InitCond.CC,InitCond.CCadj,InitCond.CC_NS,...
    InitCond.CCadj_NS,InitCond.Zroot,InitCond.B,...
    InitCond.B_NS,InitCond.HI,InitCond.HIadj,...
    InitCond.CCxAct,InitCond.CCxAct_NS,...
    InitCond.CCxW,InitCond.CCxW_NS,...
    InitCond.CCxEarlySen,InitCond.CCprev] = deal(0);
InitCond.CC0adj = Crop.CC0;
InitCond.Zroot = Crop.Zmin;
InitCond.rCor = 1;

%% Update CO2 concentration %%
% Get CO2 concentration
[Yri,~,~,~,~,~] = datevec(AOS_ClockStruct.StepStartTime);
CO2.CurrentConc = CO2.Data((CO2.Data(:,1)==Yri),2);
% Get CO2 weighting factor for first year
CO2conc = CO2.CurrentConc;
CO2ref = CO2.RefConc;
if CO2conc <= CO2ref
    fw = 0;
else
    if CO2conc >= 550
        fw = 1;
    else
        fw = 1-((550-CO2conc)/(550-CO2ref));
    end
end
% Determine initial adjustment
fCO2 = (CO2conc/CO2ref)/(1+(CO2conc-CO2ref)*((1-fw)*Crop.bsted...
    +fw*((Crop.bsted*Crop.fsink)+(Crop.bface*(1-Crop.fsink)))));
% Consider crop type
if Crop.WP >= 40
    % No correction for C4 crops
    ftype = 0;
elseif Crop.WP <= 20
    % Full correction for C3 crops
    ftype = 1;
else
    ftype = (40-Crop.WP)/(40-20);
end
% Total adjustment
Crop.fCO2 = 1+ftype*(fCO2-1);

%% Reset soil water conditions (if not running off-season) %%
if strcmp(AOS_ClockStruct.OffSeason,'N')
    % Reset water content to starting conditions
    InitCond.th = InitCond.thini;
    % Reset surface storage
    InitCond.SurfaceStorage = InitCond.SurfaceStorageIni;
end

%% Update crop parameters (if in GDD mode) %%
if Crop.CalendarType == 2
    % Extract weather data for upcoming growing season
    tSta = AOS_ClockStruct.PlantingDate(AOS_ClockStruct.SeasonCounter);
    tSto = AOS_ClockStruct.HarvestDate(AOS_ClockStruct.SeasonCounter);
    Dates = AOS_InitialiseStruct.Weather(:,1);
    StaRow = find(Dates==tSta);
    StoRow = find(Dates==tSto);
    Tmin = AOS_InitialiseStruct.Weather(StaRow:StoRow,2);
    Tmax = AOS_InitialiseStruct.Weather(StaRow:StoRow,3);

    % Calculate GDD's
    if Crop.GDDmethod == 1
        Tmean = (Tmax+Tmin)/2;
        Tmean(Tmean>Crop.Tupp) = Crop.Tupp;
        Tmean(Tmean<Crop.Tbase) = Crop.Tbase;
        GDD = Tmean-Crop.Tbase;
    elseif Crop.GDDmethod == 2
        Tmax(Tmax>Crop.Tupp) = Crop.Tupp;
        Tmax(Tmax<Crop.Tbase) = Crop.Tbase;
        Tmin(Tmin>Crop.Tupp) = Crop.Tupp;
        Tmin(Tmin<Crop.Tbase) = Crop.Tbase;
        Tmean = (Tmax+Tmin)/2;
        GDD = Tmean-Crop.Tbase;
    elseif Crop.GDDmethod == 3
        Tmax(Tmax>Crop.Tupp) = Crop.Tupp;
        Tmax(Tmax<Crop.Tbase) = Crop.Tbase;
        Tmin(Tmin>Crop.Tupp) = Crop.Tupp;
        Tmean = (Tmax+Tmin)/2;
        Tmean(Tmean<Crop.Tbase) = Crop.Tbase;
        GDD = Tmean-Crop.Tbase;
    end
    GDDcum = cumsum(GDD);

    % Find calendar days for some variables
    % 1. Calendar days from sowing to maximum canopy cover
    Crop.MaxCanopyCD = find(GDDcum > Crop.MaxCanopy,1,'first');
    % 1. Calendar days from sowing to end of vegetative growth
    Crop.CanopyDevEndCD = find(GDDcum > Crop.CanopyDevEnd,1,'first');
    % 2. Calendar days from sowing to start of yield formation
    Crop.HIstartCD = find(GDDcum > Crop.HIstart,1,'first');
    % 3. Calendar days from sowing to end of yield formation
    Crop.HIendCD = find(GDDcum > Crop.HIend,1,'first');
    % 4. Duration of yield formation in calendar days
    Crop.YldFormCD = Crop.HIendCD-Crop.HIstartCD;
    if Crop.CropType == 3
        % 1. Calendar days from sowing to end of flowering
        FloweringEnd = find(GDDcum > Crop.FloweringEnd,1,'first');
        % 2. Duration of flowering in calendar days
        Crop.FloweringCD = FloweringEnd-Crop.HIstartCD;
    end
    
    % Update harvest index growth coefficient
    Crop.HIGC = AOS_CalculateHIGC(Crop);

    % Update day to switch to linear HI build-up
    if Crop.CropType == 3
        % Determine linear switch point and HIGC rate for fruit/grain crops
        [tLin,HIGClin] = AOS_CalculateHILinear(Crop);
        Crop.tLinSwitch = tLin;
        Crop.dHILinear = HIGClin;
    else
        % No linear switch for leafy vegetable or root/tiber crops
        Crop.tLinSwitch = [];
        Crop.dHILinear = [];
    end
end

%% Update global variables %%
AOS_InitialiseStruct.InitialCondition = InitCond;
AOS_InitialiseStruct.Parameter.Crop.(CropType) = Crop;
AOS_InitialiseStruct.Parameter.CO2 = CO2;

end