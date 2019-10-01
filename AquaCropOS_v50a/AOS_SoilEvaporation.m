function [NewCond,EsAct,EsPot] = AOS_SoilEvaporation(Soil,Crop,FieldMngt,...
    InitCond,Et0,Infl,Rain,Irr,GrowingSeason,AOS_ClockStruct)
% Function to calculate daily soil evaporation in AOS


%% Store initial conditions in new structure that will be updated %%
NewCond = InitCond;

%% Initialise Wevap structure %%
Wevap = struct('Sat',0,'Fc',0,'Wp',0,'Dry',0,'Act',0);

%% Prepare stage 2 evaporation (REW gone) %%
% Only do this if it is first day of simulation, or if it is first day of
% growing season and not simulating off-season
if (AOS_ClockStruct.TimeStepCounter == 1) || ((NewCond.DAP == 1) &&...
        (strcmp(AOS_ClockStruct.OffSeason,'N')))
    % Reset storage in surface soil layer to zero
    NewCond.Wsurf = 0;
    % Set evaporation depth to minimum
    NewCond.EvapZ = Soil.EvapZmin;
    % Trigger stage 2 evaporation
    NewCond.Stage2 = true;
    % Get relative water content for start of stage 2 evaporation
    Wevap = AOS_EvapLayerWaterContent(NewCond,Soil,Wevap);
    NewCond.Wstage2 = (Wevap.Act-(Wevap.Fc-Soil.REW))/(Wevap.Sat-(Wevap.Fc-Soil.REW));
    NewCond.Wstage2 = round((100*NewCond.Wstage2))/100;
    if NewCond.Wstage2 < 0
        NewCond.Wstage2 = 0;
    end
end

%% Prepare soil evaporation stage 1 %%
% Adjust water in surface evaporation layer for any infiltration
if (Rain > 0) || ((Irr > 0))
    % Only prepare stage one when rainfall occurs, or when irrigation is
    % trigerred
    if Infl > 0
        % Update storage in surface evaporation layer for incoming
        % infiltration
        NewCond.Wsurf = Infl;
        % Water stored in surface evaporation layer cannot exceed REW
        if NewCond.Wsurf > Soil.REW;
            NewCond.Wsurf = Soil.REW;
        end
        % Reset variables
        NewCond.Wstage2 = [];
        NewCond.EvapZ = Soil.EvapZmin;
        NewCond.Stage2 = false;
    end
end

%% Calculate potential soil evaporation rate (mm/day) %%
if GrowingSeason == true
    % Adjust time for any delayed development
    if Crop.CalendarType == 1
        tAdj = NewCond.DAP-NewCond.DelayedCDs;
    elseif Crop.CalendarType == 2
        tAdj = NewCond.GDDcum-NewCond.DelayedGDDs;
    end
    
    % Calculate maximum potential soil evaporation
    EsPotMax = Soil.Kex*Et0*(1-NewCond.CCxW*(Soil.fwcc/100));
    % Calculate potential soil evaporation (given current canopy cover
    % size)
    EsPot = Soil.Kex*(1-NewCond.CCadj)*Et0;
    % Adjust potential soil evaporation for effects of withered canopy
    if (tAdj > Crop.Senescence) && (NewCond.CCxAct > 0)
        if NewCond.CC > (NewCond.CCxAct/2)
            if NewCond.CC > NewCond.CCxAct
                mult = 0;
            else
                mult = (NewCond.CCxAct-NewCond.CC)/(NewCond.CCxAct/2);       
            end
        else
            mult = 1;
        end
        EsPot = EsPot*(1-NewCond.CCxAct*(Soil.fwcc/100)*mult);
        CCxActAdj = (1.72*NewCond.CCxAct)+(NewCond.CCxAct^2)-0.3*(NewCond.CCxAct^3);
        EsPotMin = Soil.Kex*(1-CCxActAdj)*Et0;
        if EsPotMin < 0
            EsPotMin = 0;
        end
        if EsPot < EsPotMin
            EsPot = EsPotMin;
        elseif EsPot > EsPotMax
            EsPot = EsPotMax;
        end
    end
    if NewCond.PrematSenes == true
        if EsPot > EsPotMax
            EsPot = EsPotMax;
        end
    end
else
    % No canopy cover outside of growing season so potential soil
    % evaporation only depends on reference evapotranspiration
    EsPot = Soil.Kex*Et0;
end

%% Adjust potential soil evaporation for mulches %%
% Mulches
if NewCond.SurfaceStorage < 0.000001
    if FieldMngt.Mulches == 0
        % No mulches present
        EsPotMul = EsPot;
    elseif FieldMngt.Mulches == 1
        % Mulches present (percentage soil surface covered may vary
        % depending on whether within or outside growing season)
        if GrowingSeason == true
            EsPotMul = EsPot*(1-FieldMngt.fMulch*(FieldMngt.MulchPctGS/100));
        elseif GrowingSeason == false
            EsPotMul = EsPot*(1-FieldMngt.fMulch*(FieldMngt.MulchPctOS/100));
        end
    end
else
    % Surface is flooded - no adjustment of potential soil evaporation for 
    % mulches
    EsPotMul = EsPot;
end

% Assign minimum value
EsPot = min(EsPot,EsPotMul);

%% Surface evaporation %%
% Initialise actual evaporation counter
EsAct = 0;
% Evaporate surface storage
if NewCond.SurfaceStorage > 0
    if NewCond.SurfaceStorage > EsPot
        % All potential soil evaporation can be supplied by surface storage
        EsAct = EsPot;
        % Update surface storage
        NewCond.SurfaceStorage = NewCond.SurfaceStorage-EsAct;
    else
        % Surface storage is not sufficient to meet all potential soil
        % evaporation
        EsAct = NewCond.SurfaceStorage;
        % Update surface storage, evaporation layer depth, stage
        NewCond.SurfaceStorage = 0;
        NewCond.Wsurf = Soil.REW;
        NewCond.Wstage2 = [];
        NewCond.EvapZ = Soil.EvapZmin;
        NewCond.Stage2 = false;
    end
end

%% Stage 1 evaporation %%
% Determine total water to be extracted
ToExtract = EsPot-EsAct;
% Determine total water to be extracted in stage one (limited by surface
% layer water storage)
ExtractPotStg1 = min(ToExtract,NewCond.Wsurf);
% Extract water
if (ExtractPotStg1 > 0)
    % Find soil compartments covered by evaporation layer
    comp_sto = sum(Soil.Comp.dzsum<Soil.EvapZmin)+1;
    comp = 0;
    while (ExtractPotStg1 > 0) && (comp < comp_sto)
        % Increment compartment counter
        comp = comp+1;
        % Specify layer number
        layeri = Soil.Comp.Layer(comp);
        % Determine proportion of compartment in evaporation layer
        if Soil.Comp.dzsum(comp) > Soil.EvapZmin
            factor = 1-((Soil.Comp.dzsum(comp)-Soil.EvapZmin)/Soil.Comp.dz(comp));
        else
            factor = 1;
        end
        % Water storage (mm) at air dry
        Wdry = 1000*Soil.Layer.th_dry(layeri)*Soil.Comp.dz(comp);
        % Available water (mm)
        W = 1000*NewCond.th(comp)*Soil.Comp.dz(comp);
        % Water available in compartment for extraction (mm)
        AvW = (W-Wdry)*factor;
        if AvW < 0 
            AvW = 0;
        end
        if AvW >= ExtractPotStg1
            % Update actual evaporation
            EsAct = EsAct+ExtractPotStg1;
            % Update depth of water in current compartment
            W = W-ExtractPotStg1;
            % Update total water to be extracted
            ToExtract = ToExtract-ExtractPotStg1;
            % Update water to be extracted from surface layer (stage 1)
            ExtractPotStg1 = 0;
        else
            % Update actual evaporation
            EsAct = EsAct+AvW;
            % Update water to be extracted from surface layer (stage 1)
            ExtractPotStg1 = ExtractPotStg1-AvW;
            % Update total water to be extracted
            ToExtract = ToExtract-AvW;
            % Update depth of water in current compartment
            W = W-AvW;
        end
        % Update water content
        NewCond.th(comp) = W/(1000*Soil.Comp.dz(comp));
    end
    
    % Update surface evaporation layer water balance
    NewCond.Wsurf = NewCond.Wsurf-EsAct;
    if (NewCond.Wsurf < 0) || (ExtractPotStg1 > 0.0001)
        NewCond.Wsurf = 0;
    end
    
    % If surface storage completely depleted, prepare stage 2
    if NewCond.Wsurf < 0.0001
        % Get water contents (mm)
        Wevap = AOS_EvapLayerWaterContent(NewCond,Soil,Wevap);
        % Proportional water storage for start of stage two evaporation
        NewCond.Wstage2 = (Wevap.Act-(Wevap.Fc-Soil.REW))/(Wevap.Sat-(Wevap.Fc-Soil.REW));
        NewCond.Wstage2 = round((100*NewCond.Wstage2))/100;
        if NewCond.Wstage2 < 0
            NewCond.Wstage2 = 0;
        end
    end
end

%% Stage 2 evaporation %%
% Extract water
if ToExtract > 0 
    % Start stage 2
    NewCond.Stage2 = true;
    % Get sub-daily evaporative demand
    Edt = ToExtract/AOS_ClockStruct.EvapTimeSteps;
    % Loop sub-daily steps
    for jj = 1:AOS_ClockStruct.EvapTimeSteps
        % Get current water storage (mm)
        Wevap = AOS_EvapLayerWaterContent(NewCond,Soil,Wevap);
        % Get water storage (mm) at start of stage 2 evaporation
        Wupper = NewCond.Wstage2*(Wevap.Sat-(Wevap.Fc-Soil.REW))+(Wevap.Fc-Soil.REW);
        % Get water storage (mm) when there is no evaporation
        Wlower = Wevap.Dry;
        % Get relative depletion of evaporation storage in stage 2
        Wrel = (Wevap.Act-Wlower)/(Wupper-Wlower);
        % Check if need to expand evaporation layer
        if Soil.EvapZmax > Soil.EvapZmin
            Wcheck = Soil.fWrelExp*((Soil.EvapZmax-NewCond.EvapZ)/(Soil.EvapZmax-Soil.EvapZmin));
            while (Wrel < Wcheck) && (NewCond.EvapZ < Soil.EvapZmax)
                % Expand evaporation layer by 1 mm
                NewCond.EvapZ = NewCond.EvapZ+0.001;
                % Update water storage (mm) in evaporation layer
                Wevap = AOS_EvapLayerWaterContent(NewCond,Soil,Wevap);
                Wupper = NewCond.Wstage2*(Wevap.Sat-(Wevap.Fc-Soil.REW))+(Wevap.Fc-Soil.REW);
                Wlower = Wevap.Dry;
                % Update relative depletion of evaporation storage
                Wrel = (Wevap.Act-Wlower)/(Wupper-Wlower);
                Wcheck = Soil.fWrelExp*((Soil.EvapZmax-NewCond.EvapZ)/(Soil.EvapZmax-Soil.EvapZmin));
            end
        end
        % Get stage 2 evaporation reduction coefficient
        Kr = (exp(Soil.fevap*Wrel)-1)/(exp(Soil.fevap)-1);
        if Kr > 1
            Kr = 1;
        end
        % Get water to extract (mm)
        ToExtractStg2 = Kr*Edt;

        % Extract water from compartments
        comp_sto = sum(Soil.Comp.dzsum<NewCond.EvapZ)+1;
        comp = 0;     
        while (ToExtractStg2 > 0) && (comp < comp_sto)
            % Increment compartment counter
            comp = comp+1;
            % Specify layer number
            layeri = Soil.Comp.Layer(comp);
            % Determine proportion of compartment in evaporation layer
            if Soil.Comp.dzsum(comp) > NewCond.EvapZ
                factor = 1-((Soil.Comp.dzsum(comp)-NewCond.EvapZ)/Soil.Comp.dz(comp));
            else
                factor = 1;
            end          
            % Water storage (mm) at air dry
            Wdry = 1000*Soil.Layer.th_dry(layeri)*Soil.Comp.dz(comp);
            % Available water (mm)
            W = 1000*NewCond.th(comp)*Soil.Comp.dz(comp);
            % Water available in compartment for extraction (mm)
            AvW = (W-Wdry)*factor;
            if AvW >= ToExtractStg2
                % Update actual evaporation
                EsAct = EsAct+ToExtractStg2;
                % Update depth of water in current compartment
                W = W-ToExtractStg2;
                % Update total water to be extracted
                ToExtract = ToExtract-ToExtractStg2;
                % Update water to be extracted from surface layer (stage 1)
                ToExtractStg2 = 0; 
            else
                % Update actual evaporation
                EsAct = EsAct+AvW;
                % Update depth of water in current compartment
                W = W-AvW;
                % Update water to be extracted from surface layer (stage 1)
                ToExtractStg2 = ToExtractStg2-AvW;
                % Update total water to be extracted
                ToExtract = ToExtract-AvW;                
            end
            % Update water content
            NewCond.th(comp) = W/(1000*Soil.Comp.dz(comp));           
        end 
    end   
end

%% Store potential evaporation for irrigation calculations on next day %%
NewCond.Epot = EsPot;

end