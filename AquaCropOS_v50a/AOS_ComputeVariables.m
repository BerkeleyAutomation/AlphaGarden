function [ParamStruct] = AOS_ComputeVariables(ParamStruct,Weather,...
    ClockStruct,GwStruct,CropChoices,FileLocation)
% Function to compute additional variables needed to run AOS

%% Compute water contents and saturated hydraulic conductivity %%
if ParamStruct.Soil.CalcSHP == 0
    % Read soil texture file
    filename = strcat(FileLocation.Input,FileLocation.SoilHydrologyFilename);
    fileID = fopen(filename);
    if fileID == -1
        % Can't find text file defining soil hydraulic properties
        % Throw error message
        fprintf(2,'Error - Soil hydrology input file not found\n');
    end
    % Load data
    Data = textscan(fileID,'%f %f %f %f %f %f','delimiter','\t','headerlines',2);
    fclose(fileID);
    % Assign data
    ParamStruct.Soil.Layer.dz = Data{1,2}(:)';
    ParamStruct.Soil.Layer.th_s = Data{1,3}(:)';
    ParamStruct.Soil.Layer.th_fc = Data{1,4}(:)';
    ParamStruct.Soil.Layer.th_wp = Data{1,5}(:)';
    ParamStruct.Soil.Layer.Ksat = Data{1,6}(:)';
    % Calculate additional variables
    ParamStruct.Soil.Layer.th_dry = ParamStruct.Soil.Layer.th_wp/2;    
elseif ParamStruct.Soil.CalcSHP == 1
    % Read soil texture file
    filename = strcat(FileLocation.Input,FileLocation.SoilTextureFilename);
    fileID = fopen(filename);
    if fileID == -1
        % Can't find text file defining soil textural properties
        % Throw error message
        fprintf(2,'Error - Soil texture input file not found\n');
    end
    % Load data
    Data = textscan(fileID,'%f %f %f %f %f %f','delimiter','\t','headerlines',2);
    fclose(fileID);
    % Create soil dz vector
    ParamStruct.Soil.Layer.dz = Data{1,2}(:)';
    ParamStruct.Soil.Layer.Sand = (Data{1,3}(:)')/100;
    ParamStruct.Soil.Layer.Clay = (Data{1,4}(:)')/100;
    ParamStruct.Soil.Layer.OrgMat = Data{1,5}(:)';
    ParamStruct.Soil.Layer.DF = Data{1,6}(:)';
    % Calculate soil hydraulic properties using pedotransfer function
    % method (Saxton et al., 2006)
    [thdry,thwp,thfc,ths,ksat] = AOS_SoilHydraulicProperties(ParamStruct.Soil);
    ParamStruct.Soil.Layer.th_dry = thdry;
    ParamStruct.Soil.Layer.th_wp = thwp;
    ParamStruct.Soil.Layer.th_fc = thfc;
    ParamStruct.Soil.Layer.th_s = ths;
    ParamStruct.Soil.Layer.Ksat = ksat;
end

%% Assign field capacity values to each soil compartment %%
for ii = 1:ParamStruct.Soil.nComp
    layeri = ParamStruct.Soil.Comp.Layer(ii);
    ParamStruct.Soil.Comp.th_fc(ii) = ParamStruct.Soil.Layer.th_fc(layeri);
end

%% Calculate capillary rise parameters for all soil layers %%
% Only do calculation if water table is present. Calculations use equations 
% described in Raes et al. (2012)
if GwStruct.WaterTable == 1
    aCR = zeros(1,ParamStruct.Soil.nLayer);
    bCR = zeros(1,ParamStruct.Soil.nLayer);
    for ii = 1:ParamStruct.Soil.nLayer
        thwp = ParamStruct.Soil.Layer.th_wp(ii);
        thfc = ParamStruct.Soil.Layer.th_fc(ii);
        ths = ParamStruct.Soil.Layer.th_s(ii);
        Ksat = ParamStruct.Soil.Layer.Ksat(ii);
        if (thwp >= 0.04) && (thwp <= 0.15) && (thfc >= 0.09) &&...
                (thfc <= 0.28) && (ths >= 0.32) && (ths <= 0.51)
            % Sandy soil class
            if (Ksat >= 200) && (Ksat <= 2000)
                aCR(ii) = -0.3112-(Ksat*(10^-5));
                bCR(ii) = -1.4936+(0.2416*log(Ksat));
            elseif Ksat < 200
                aCR(ii) = -0.3112-(200*(10^-5));
                bCR(ii) = -1.4936+(0.2416*log(200));
            elseif Ksat > 2000
                aCR(ii) = -0.3112-(2000*(10^-5));
                bCR(ii) = -1.4936+(0.2416*log(2000));
            end 
        elseif (thwp >= 0.06) && (thwp <= 0.20) && (thfc >= 0.23) &&...
                (thfc <= 0.42) && (ths >= 0.42) && (ths <= 0.55)
            % Loamy soil class
            if (Ksat >= 100) && (Ksat <= 750)
                aCR(ii) = -0.4986+(9*(10^-5)*Ksat);
                bCR(ii) = -2.132+(0.4778*log(Ksat));
            elseif Ksat < 100
                aCR(ii) = -0.4986+(9*(10^-5)*100);
                bCR(ii) = -2.132+(0.4778*log(100));
            elseif Ksat > 750
                aCR(ii) = -0.4986+(9*(10^-5)*750);
                bCR(ii) = -2.132+(0.4778*log(750));
            end 
        elseif (thwp >= 0.16) && (thwp <= 0.34) && (thfc >= 0.25) &&...
                (thfc <= 0.45) && (ths >= 0.40) && (ths <= 0.53)
            % Sandy clayey soil class
            if (Ksat >= 5) && (Ksat <= 150)
                aCR(ii) = -0.5677-(4*(10^-5)*Ksat);
                bCR(ii) = -3.7189+(0.5922*log(Ksat));
            elseif Ksat < 5
                aCR(ii) = -0.5677-(4*(10^-5)*5);
                bCR(ii) = -3.7189+(0.5922*log(5));
            elseif Ksat > 150
                aCR(ii) = -0.5677-(4*(10^-5)*150);
                bCR(ii) = -3.7189+(0.5922*log(150));
            end 
        elseif (thwp >= 0.20) && (thwp <= 0.42) && (thfc >= 0.40) &&...
                (thfc <= 0.58) && (ths >= 0.49) && (ths <= 0.58)
            % Silty clayey soil class
            if (Ksat >= 1) && (Ksat <= 150)
                aCR(ii) = -0.6366+(8*(10^-4)*Ksat);
                bCR(ii) = -1.9165+(0.7063*log(Ksat));
            elseif Ksat < 1
                aCR(ii) = -0.6366+(8*(10^-4)*1);
                bCR(ii) = -1.9165+(0.7063*log(1));
            elseif Ksat > 150
                aCR(ii) = -0.6366+(8*(10^-4)*150);
                bCR(ii) = -1.9165+(0.7063*log(150));
            end    
        end
    end
    ParamStruct.Soil.Layer.aCR = aCR;
    ParamStruct.Soil.Layer.bCR = bCR;    
end 

%% Calculate drainage characteristic (tau) %%
% Calculations use equation given by Raes et al. 2012
for ii = 1:ParamStruct.Soil.nLayer
    ParamStruct.Soil.Layer.tau(ii) = 0.0866*(ParamStruct.Soil.Layer.Ksat(ii)^0.35);
    ParamStruct.Soil.Layer.tau(ii) = round((100*ParamStruct.Soil.Layer.tau(ii)))/100;
    if ParamStruct.Soil.Layer.tau(ii) > 1
       ParamStruct.Soil.Layer.tau(ii) = 1;
    elseif ParamStruct.Soil.Layer.tau(ii) < 0
        ParamStruct.Soil.Layer.tau(ii) = 0;
    end
end

%% Calculate readily evaporable water in surface layer %%
if ParamStruct.Soil.AdjREW == 0
    ParamStruct.Soil.REW = round((1000*(ParamStruct.Soil.Layer.th_fc(1)-...
        ParamStruct.Soil.Layer.th_dry(1))*ParamStruct.Soil.EvapZsurf));
end

%% Calculate upper and lower curve numbers %%
ParamStruct.Soil.CNbot = round(1.4*(exp(-14*log(10)))+(0.507*ParamStruct.Soil.CN)-...
    (0.00374*ParamStruct.Soil.CN^2)+(0.0000867*ParamStruct.Soil.CN^3));
ParamStruct.Soil.CNtop = round(5.6*(exp(-14*log(10)))+(2.33*ParamStruct.Soil.CN)-...
    (0.0209*ParamStruct.Soil.CN^2)+(0.000076*ParamStruct.Soil.CN^3));

%% Fit function relating water content to curve number %%
% Use properties of top soil layer
xi = [ParamStruct.Soil.Layer.th_wp(1),((ParamStruct.Soil.Layer.th_fc(1)+...
    ParamStruct.Soil.Layer.th_wp(1))/2),ParamStruct.Soil.Layer.th_fc(1)];
yi = [ParamStruct.Soil.CNbot,ParamStruct.Soil.CN,ParamStruct.Soil.CNtop];
ParamStruct.Soil.CNf = pchip(xi,yi);

%% Calculate additional parameters for all crop types in mix %%
CropNames = fieldnames(ParamStruct.Crop);
nCrops = size(CropNames,1);
for ii = 1:nCrops
    % Fractional canopy cover size at emergence 
    ParamStruct.Crop.(CropNames{ii}).CC0 = round(10000*...
        (ParamStruct.Crop.(CropNames{ii}).PlantPop*...
        ParamStruct.Crop.(CropNames{ii}).SeedSize)*10^-8)/10000;
    % Root extraction terms 
    SxTopQ = ParamStruct.Crop.(CropNames{ii}).SxTopQ;
    SxBotQ = ParamStruct.Crop.(CropNames{ii}).SxBotQ;
    S1 = ParamStruct.Crop.(CropNames{ii}).SxTopQ;
    S2 = ParamStruct.Crop.(CropNames{ii}).SxBotQ;
    if S1 == S2
        SxTop = S1;
        SxBot = S2;
    else
        if SxTopQ < SxBotQ
            S1 = SxBotQ;
            S2 = SxTopQ;
        end
        xx = 3*(S2/(S1-S2));
        if xx < 0.5
            SS1 = (4/3.5)*S1;
            SS2 = 0;
        else
            SS1 = (xx+3.5)*(S1/(xx+3));
            SS2 = (xx-0.5)*(S2/xx);
        end
        if SxTopQ > SxBotQ
            SxTop = SS1;
            SxBot = SS2;
        else
            SxTop = SS2;
            SxBot = SS1;
        end
    end
    ParamStruct.Crop.(CropNames{ii}).SxTop = SxTop;
    ParamStruct.Crop.(CropNames{ii}).SxBot = SxBot;
    
    % Water stress thresholds
    ParamStruct.Crop.(CropNames{ii}).p_up = [ParamStruct.Crop.(CropNames{ii}).p_up1,...
        ParamStruct.Crop.(CropNames{ii}).p_up2,ParamStruct.Crop.(CropNames{ii}).p_up3,...
        ParamStruct.Crop.(CropNames{ii}).p_up4];
    ParamStruct.Crop.(CropNames{ii}).p_lo = [ParamStruct.Crop.(CropNames{ii}).p_lo1,...
        ParamStruct.Crop.(CropNames{ii}).p_lo2,ParamStruct.Crop.(CropNames{ii}).p_lo3,...
        ParamStruct.Crop.(CropNames{ii}).p_lo4];
    ParamStruct.Crop.(CropNames{ii}).fshape_w = [ParamStruct.Crop.(CropNames{ii}).fshape_w1,...
        ParamStruct.Crop.(CropNames{ii}).fshape_w2,ParamStruct.Crop.(CropNames{ii}).fshape_w3,...
        ParamStruct.Crop.(CropNames{ii}).fshape_w4];
    fields = {'p_up1','p_up2','p_up3','p_up4','p_lo1','p_lo2','p_lo3',...
        'p_lo4','fshape_w1','fshape_w2','fshape_w3','fshape_w4'};
    ParamStruct.Crop.(CropNames{ii}) = rmfield(ParamStruct.Crop.(CropNames{ii}),fields);
    
    % Flowering function 
    if ParamStruct.Crop.(CropNames{ii}).CropType == 3
        ParamStruct.Crop.(CropNames{ii}).flowerfun =...
            @(xx) (0.00558*(xx.^0.63))-(0.000969*xx)-0.00383;
    end
    
    % Crop calendar
    ParamStruct.Crop.(CropNames{ii}) =...
        AOS_ComputeCropCalendar(ParamStruct.Crop.(CropNames{ii}),CropNames{ii},...
        CropChoices,Weather);

    % Harvest index growth coefficient
    ParamStruct.Crop.(CropNames{ii}).HIGC =...
        AOS_CalculateHIGC(ParamStruct.Crop.(CropNames{ii}));
    
    % Days to linear HI switch point
    if ParamStruct.Crop.(CropNames{ii}).CropType == 3
        % Determine linear switch point and HIGC rate for fruit/grain crops
        [tLin,HIGClin] = AOS_CalculateHILinear(ParamStruct.Crop.(CropNames{ii}));
        ParamStruct.Crop.(CropNames{ii}).tLinSwitch = tLin;
        ParamStruct.Crop.(CropNames{ii}).dHILinear = HIGClin;
    else
        % No linear switch for leafy vegetable or root/tiber crops
        ParamStruct.Crop.(CropNames{ii}).tLinSwitch = [];
        ParamStruct.Crop.(CropNames{ii}).dHILinear = [];
    end
end
   
%% Calculate WP adjustment factor for elevation in CO2 concentration %%
% Load CO2 data 
filename = strcat(FileLocation.Input,FileLocation.CO2Filename);
fileID = fopen(filename);
if fileID == -1
    % Can't find text file defining CO2 concentrations
    % Throw error message
    fprintf(2,'Error - CO2 input file not found\n');
end
CO2Data = textscan(fileID,'%f %f','delimiter',':','commentstyle','%%');
fclose(fileID);
% Years 
Yrs = CO2Data{1,1}(:);
% CO2 concentrations (ppm)
CO2 = CO2Data{1,2}(:);
% Interpolate data
[StaYr,~,~,~,~,~] = datevec(ClockStruct.SimulationStartDate);
[EndYr,~,~,~,~,~] = datevec(ClockStruct.SimulationEndDate);
YrsVec = StaYr:EndYr;
CO2conc = interp1(Yrs,CO2,YrsVec);
% Store data
ParamStruct.CO2.Data = [YrsVec',CO2conc'];

% Define reference CO2 concentration
ParamStruct.CO2.RefConc = 369.41;

% Get CO2 concentration for first year
[Yri,~,~,~,~,~] = datevec(ClockStruct.SimulationStartDate);
ParamStruct.CO2.CurrentConc = ParamStruct.CO2.Data((ParamStruct.CO2.Data(:,1)==Yri),2);

% Get CO2 weighting factor for first year
CO2ref = ParamStruct.CO2.RefConc;
CO2conc = ParamStruct.CO2.CurrentConc;
if CO2conc <= CO2ref
    fw = 0;
else
    if CO2conc >= 550
        fw = 1;
    else
        fw = 1-((550-CO2conc)/(550-CO2ref));
    end
end

% Determine adjustment for each crop in first year of simulation
for ii = 1:nCrops
% Determine initial adjustment
fCO2 = (CO2conc/CO2ref)/(1+(CO2conc-CO2ref)*((1-fw)*...
    ParamStruct.Crop.(CropNames{ii}).bsted+fw*((ParamStruct.Crop.(CropNames{ii}).bsted*...
    ParamStruct.Crop.(CropNames{ii}).fsink)+(ParamStruct.Crop.(CropNames{ii}).bface*...
    (1-ParamStruct.Crop.(CropNames{ii}).fsink)))));
% Consider crop type
if ParamStruct.Crop.(CropNames{ii}).WP >= 40
    % No correction for C4 crops
    ftype = 0;
elseif ParamStruct.Crop.(CropNames{ii}).WP <= 20
    % Full correction for C3 crops
    ftype = 1;
else
    ftype = (40-ParamStruct.Crop.(CropNames{ii}).WP)/(40-20);
end
% Total adjustment
ParamStruct.Crop.(CropNames{ii}).fCO2 = 1+ftype*(fCO2-1);
end

end

