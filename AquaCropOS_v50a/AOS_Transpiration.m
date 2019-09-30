function [TrAct,TrPot_NS,TrPot0,NewCond,IrrNet] = AOS_Transpiration(Soil,...
    Crop,IrrMngt,InitCond,Et0,CO2,GrowingSeason)
% Function to calculate crop transpiration on current day

%% Store initial conditions %%
NewCond = InitCond;

%% Calculate transpiration (if in growing season) %%
if GrowingSeason == true
    %% Calculate potential transpiration %%
    % 1. No prior water stress
    % Update ageing days counter
    if NewCond.DAP > Crop.MaxCanopyCD
        NewCond.AgeDays_NS = NewCond.DAP-Crop.MaxCanopyCD;
    end
    % Update crop coefficient for ageing of canopy
    if NewCond.AgeDays_NS > 5
        Kcb_NS = Crop.Kcb-((NewCond.AgeDays_NS-5)*(Crop.fage/100))*NewCond.CCxW_NS;
    else
        Kcb_NS = Crop.Kcb;
    end
    % Update crop coefficient for CO2 concentration
    if CO2.CurrentConc > CO2.RefConc
        Kcb_NS = Kcb_NS*(1-0.05*((CO2.CurrentConc-CO2.RefConc)/(550-CO2.RefConc)));
    end
    % Determine potential transpiration rate (no water stress)
    TrPot_NS = Kcb_NS*(NewCond.CCadj_NS)*Et0;
    
    % 2. Potential prior water stress and/or delayed development
    % Update ageing days counter
    DAPadj = NewCond.DAP-NewCond.DelayedCDs;
    if DAPadj > Crop.MaxCanopyCD
        NewCond.AgeDays = DAPadj-Crop.MaxCanopyCD;
    end
    % Update crop coefficient for ageing of canopy
    if NewCond.AgeDays > 5
        Kcb = Crop.Kcb-((NewCond.AgeDays-5)*(Crop.fage/100))*NewCond.CCxW;
    else
        Kcb = Crop.Kcb;
    end
    % Update crop coefficient for CO2 concentration
    if CO2.CurrentConc > CO2.RefConc
        Kcb = Kcb*(1-0.05*((CO2.CurrentConc-CO2.RefConc)/(550-CO2.RefConc)));
    end
    % Determine potential transpiration rate
    TrPot0 = Kcb*(NewCond.CCadj)*Et0;
    % Correct potential transpiration for dying green canopy effects
    if NewCond.CC < NewCond.CCxW
        if (NewCond.CCxW > 0.001) && (NewCond.CC > 0.001)
            TrPot0 = TrPot0*((NewCond.CC/NewCond.CCxW)^Crop.a_Tr);
        end
    end
    
    %% Calculate surface layer transpiration %%
    if (NewCond.SurfaceStorage > 0) && (NewCond.DaySubmerged < Crop.LagAer)
        % Update submergence days counter
        NewCond.DaySubmerged = NewCond.DaySubmerged+1;
        % Update anerobic conditions counter for each compartment
        for ii = 1:Soil.nComp
            % Increment aeration days counter for compartment ii
            NewCond.AerDaysComp(ii) = NewCond.AerDaysComp(ii)+1;
            if NewCond.AerDaysComp(ii) > Crop.LagAer
                NewCond.AerDaysComp(ii) = Crop.LagAer;
            end
        end
        % Reduce actual transpiration that is possible to account for
        % aeration stress due to extended submergence
        fSub = 1-(NewCond.DaySubmerged/Crop.LagAer);
        if NewCond.SurfaceStorage > (fSub*TrPot0)
            % Transpiration occurs from surface storage
            NewCond.SurfaceStorage = NewCond.SurfaceStorage-(fSub*TrPot0);
            TrAct0 = fSub*TrPot0;
        else
            % No transpiration from surface storage
            TrAct0 = 0;
        end
        if TrAct0 < (fSub*TrPot0)
            % More water can be extracted from soil profile for transpiration
            TrPot = (fSub*TrPot0)-TrAct0;
        else
            % No more transpiration possible on current day
            TrPot = 0;
        end
    else
        % No surface transpiration occurs
        TrPot = TrPot0;
        TrAct0 = 0;
    end
    
    %% Update potential root zone transpiration for water stress %%
    % Determine root zone water content
    [~,Dr,TAW,thRZ] = AOS_RootZoneWater(Soil,Crop,NewCond);
    % Calculate water stress coefficients
    beta = true;
    Ksw = AOS_WaterStress(Crop,NewCond,Dr,TAW,Et0,beta);
    % Calculate aeration stress coefficients
    [Ksa,NewCond] = AOS_AerationStress(Crop,NewCond,thRZ);
    % Maximum stress effect
    Ks = min(Ksw.StoLin,Ksa.Aer);
    % Update potential transpiration in root zone
    if IrrMngt.IrrMethod ~= 4
        % No adjustment to TrPot for water stress when in net irrigation mode
        TrPot = TrPot*Ks;
    end
    
    %% Determine compartments covered by root zone %%
    % Compartments covered by the root zone
    rootdepth = max(NewCond.Zroot,Crop.Zmin);
    rootdepth = round((100*rootdepth))/100;
    comp_sto = sum(Soil.Comp.dzsum<rootdepth)+1;
    RootFact = zeros(1,Soil.nComp);
    % Determine fraction of each compartment covered by root zone
    for ii = 1:comp_sto
        if Soil.Comp.dzsum(ii) > rootdepth
            RootFact(ii) = 1-((Soil.Comp.dzsum(ii)-rootdepth)/Soil.Comp.dz(ii));
        else
            RootFact(ii) = 1;
        end
    end
    
    %% Determine maximum sink term for each compartment %%
    SxComp = zeros(1,Soil.nComp);
    if IrrMngt.IrrMethod == 4
        % Net irrigation mode
        for ii = 1:comp_sto
            SxComp(ii) = (Crop.SxTop+Crop.SxBot)/2;
        end
    else
        % Maximum sink term declines linearly with depth
        SxCompBot = Crop.SxTop;
        for ii = 1:comp_sto
            SxCompTop = SxCompBot;
            if Soil.Comp.dzsum(ii) <= rootdepth
                SxCompBot = Crop.SxBot*NewCond.rCor+((Crop.SxTop-Crop.SxBot*...
                    NewCond.rCor)*((rootdepth-Soil.Comp.dzsum(ii))/rootdepth));
            else
                SxCompBot = Crop.SxBot*NewCond.rCor;
            end
            SxComp(ii) = (SxCompTop+SxCompBot)/2;
        end
    end
    
    %% Extract water %%
    ToExtract = TrPot;
    comp = 0;
    TrAct = 0;
    while (ToExtract > 0) && (comp < comp_sto)
        % Increment compartment
        comp = comp+1;
        % Specify layer number
        layeri = Soil.Comp.Layer(comp);
        
        % Determine TAW (m3/m3) for compartment
        thTAW = Soil.Layer.th_fc(layeri)-Soil.Layer.th_wp(layeri);
        if Crop.ETadj == 1
            % Adjust stomatal stress threshold for Et0 on current day
            p_up_sto = Crop.p_up(2)+(0.04*(5-Et0)).*(log10(10-9*Crop.p_up(2)));
        end
        % Determine critical water content at which stomatal closure will
        % occur in compartment
        thCrit = Soil.Layer.th_fc(layeri)-(thTAW*p_up_sto);
        
        % Check for soil water stress
        if NewCond.th(comp) >= thCrit
            % No water stress effects on transpiration
            KsComp = 1;
        elseif NewCond.th(comp) > Soil.Layer.th_wp(layeri)
            % Transpiration from compartment is affected by water stress
            Wrel = (Soil.Layer.th_fc(layeri)-NewCond.th(comp))/...
                (Soil.Layer.th_fc(layeri)-Soil.Layer.th_wp(layeri));
            pRel = (Wrel-Crop.p_up(2))/(Crop.p_lo(2)-Crop.p_up(2));
            if pRel <= 0
                KsComp = 1;
            elseif pRel >= 1
                KsComp = 0;
            else
                KsComp = 1-((exp(pRel*Crop.fshape_w(2))-1)/(exp(Crop.fshape_w(2))-1));
            end
            if KsComp > 1
                KsComp = 1;
            elseif KsComp < 0
                KsComp = 0;
            end
        else
            % No transpiration is possible from compartment as water
            % content does not exceed wilting point
            KsComp = 0;
        end
        
        % Adjust compartment stress factor for aeration stress
        if NewCond.DaySubmerged >= Crop.LagAer
            % Full aeration stress - no transpiration possible from
            % compartment
            AerComp = 0;
        elseif NewCond.th(comp) > (Soil.Layer.th_s(layeri)-(Crop.Aer/100))
            % Increment aeration stress days counter
            NewCond.AerDaysComp(comp) = NewCond.AerDaysComp(comp)+1;
            if NewCond.AerDaysComp(comp) >= Crop.LagAer
                NewCond.AerDaysComp(comp) = Crop.LagAer;
                fAer = 0;
            else
                fAer = 1;
            end
            % Calculate aeration stress factor
            AerComp = (Soil.Layer.th_s(layeri)-NewCond.th(comp))/...
                (Soil.Layer.th_s(layeri)-(Soil.Layer.th_s(layeri)-(Crop.Aer/100)));
            if AerComp < 0
                AerComp = 0;
            end
            AerComp = (fAer+(NewCond.AerDaysComp(comp)-1)*AerComp)/...
                (fAer+NewCond.AerDaysComp(comp)-1);
        else
            % No aeration stress as number of submerged days does not
            % exceed threshold for initiation of aeration stress
            AerComp = 1;
            NewCond.AerDaysComp(comp) = 0;
        end
        
        % Extract water
        ThToExtract = (ToExtract/1000)/Soil.Comp.dz(comp);
        if IrrMngt.IrrMethod == 4
            % Don't reduce compartment sink for stomatal water stress if in
            % net irrigation mode. Stress only occurs due to deficient
            % aeration conditions
            Sink = AerComp*SxComp(comp)*RootFact(comp);
        else
            % Reduce compartment sink for greatest of stomatal and aeration
            % stress
            if KsComp == AerComp
                Sink = KsComp*SxComp(comp)*RootFact(comp);
            else
                Sink = min(KsComp,AerComp)*SxComp(comp)*RootFact(comp);
            end
        end
        
        % Limit extraction to demand
        if ThToExtract < Sink
            Sink = ThToExtract;
        end
        
        % Limit extraction to avoid compartment water content dropping
        % below air dry
        if (InitCond.th(comp)-Sink) < Soil.Layer.th_dry(layeri)
            Sink = InitCond.th(comp)-Soil.Layer.th_dry(layeri);
            if Sink < 0
                Sink = 0;
            end
        end
        % Update water content in compartment
        NewCond.th(comp) = InitCond.th(comp)-Sink;
        % Update amount of water to extract
        ToExtract = ToExtract-(Sink*1000*Soil.Comp.dz(comp));
        % Update actual transpiration
        TrAct = TrAct+(Sink*1000*Soil.Comp.dz(comp));
    end
    
    %% Add net irrigation water requirement (if this mode is specified) %%
    if (IrrMngt.IrrMethod == 4) && (TrPot > 0)
        % Initialise net irrigation counter
        IrrNet = 0;
        % Get root zone water content
        [~,~,~,thRZ] = AOS_RootZoneWater(Soil,Crop,NewCond);
        % Determine critical water content for net irrigation
        thCrit = thRZ.Wp+((IrrMngt.NetIrrSMT/100)*(thRZ.Fc-thRZ.Wp));
        % Check if root zone water content is below net irrigation trigger
        if thRZ.Act < thCrit
            % Initialise layer counter
            prelayer = 0;
            for ii = 1:comp_sto
                % Get soil layer
                layeri = Soil.Comp.Layer(ii);
                if layeri > prelayer
                    % If in new layer, update critical water content for
                    % net irrigation
                    thCrit = Soil.Layer.th_wp(layeri)+((IrrMngt.NetIrrSMT/100)*...
                        (Soil.Layer.th_fc(layeri)-Soil.Layer.th_wp(layeri)));
                    % Update layer counter
                    prelayer = layeri;
                end
                % Determine necessary change in water content in
                % compartments to reach critical water content
                dWC = RootFact(ii)*(thCrit-NewCond.th(ii))*1000*Soil.Comp.dz(ii);
                % Update water content
                NewCond.th(ii) = NewCond.th(ii)+(dWC/(1000*Soil.Comp.dz(ii)));
                % Update net irrigation counter
                IrrNet = IrrNet+dWC;
            end
        end
        % Update net irrigation counter for the growing season
        NewCond.IrrNetCum = NewCond.IrrNetCum+IrrNet;
    elseif (IrrMngt.IrrMethod == 4) && (TrPot <= 0)
        % No net irrigation as potential transpiration is zero
        IrrNet = 0;
    else
        % No net irrigation as not in net irrigation mode
        IrrNet = 0;
        NewCond.IrrNetCum = 0;
    end
    
    %% Add any surface transpiration to root zone total %%
    TrAct = TrAct+TrAct0;
    
    %% Feedback with canopy cover development %%
    % If actual transpiration is zero then no canopy cover growth can occur
    if ((NewCond.CC-NewCond.CCprev) > 0.005) && (TrAct == 0)
        NewCond.CC = NewCond.CCprev;
    end
    
    %% Update transpiration ratio %%
    if TrPot0 > 0
        if TrAct < TrPot0
            NewCond.TrRatio = TrAct/TrPot0;
        else
            NewCond.TrRatio = 1;
        end
    else
        NewCond.TrRatio = 1;
    end
    if NewCond.TrRatio < 0
        NewCond.TrRatio = 0;
    elseif NewCond.TrRatio > 1
        NewCond.TrRatio = 1;
    end
else
    % No transpiration if not in growing season
    TrAct = 0;
    TrPot0 = 0;
    TrPot_NS = 0;
    % No irrigation if not in growing season
    IrrNet = 0;
    NewCond.IrrNetCum = 0;
end

%% Store potential transpiration for irrigation calculations on next day %%
NewCond.Tpot = TrPot0;

end
