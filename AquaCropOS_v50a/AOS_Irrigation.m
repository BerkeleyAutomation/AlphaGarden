function [NewCond,Irr] = AOS_Irrigation(InitCond,IrrMngt,Crop,Soil,...
    AOS_ClockStruct,GrowingSeason,Rain,Runoff)
% Function to get irrigation depth for current day

%% Store intial conditions for updating %%
NewCond = InitCond;

%% Calculate root zone water content and depletion %%
[~,Dr,TAW,thRZ] = AOS_RootZoneWater(Soil,Crop,NewCond);

%% Determine adjustment for inflows and outflows on current day %%
if thRZ.Act > thRZ.Fc
    rootdepth = max(InitCond.Zroot,Crop.Zmin);
    AbvFc = (thRZ.Act-thRZ.Fc)*1000*rootdepth;
else
    AbvFc = 0;
end  
WCadj = InitCond.Tpot+InitCond.Epot-Rain+Runoff-AbvFc;
        
%% Determine irrigation depth (mm/day) to be applied %%
if GrowingSeason == true
    % Update growth stage if it is first day of a growing season
    if NewCond.DAP == 1
        NewCond.GrowthStage = 1;
    end
    % Run irrigation depth calculation
    if IrrMngt.IrrMethod == 0 % Rainfed - no irrigation
        Irr = 0;
    elseif IrrMngt.IrrMethod == 1 % Irrigation - soil moisture
        % Get soil moisture target for current growth stage
        SMT = IrrMngt.SMT(NewCond.GrowthStage);
        % Determine threshold to initiate irrigation
        IrrThr = (1-SMT/100)*TAW;
        % Adjust depletion for inflows and outflows today
        Dr = Dr+WCadj;
        if Dr < 0
            Dr = 0;
        end
        % Check if depletion exceeds threshold
        if Dr > IrrThr
            % Irrigation will occur
            IrrReq = max(0,Dr);
            % Adjust irrigation requirements for application efficiency
            EffAdj = ((100-IrrMngt.AppEff)+100)/100;
            IrrReq = IrrReq*EffAdj;
            % Limit irrigation to maximum depth
            Irr = min(IrrMngt.MaxIrr,IrrReq);
        else
            % No irrigation
            Irr = 0;
        end
    elseif IrrMngt.IrrMethod == 2 % Irrigation - fixed interval
        % Get number of days in growing season so far (subtract 1 so that
        % always irrigate first on day 1 of each growing season)
        nDays = NewCond.DAP-1;
        % Adjust depletion for inflows and outflows today
        Dr = Dr+WCadj;
        if Dr < 0
            Dr = 0;
        end
        if rem(nDays,IrrMngt.IrrInterval) == 0
            % Irrigation occurs
            IrrReq = max(0,Dr);
            % Adjust irrigation requirements for application efficiency
            EffAdj = ((100-IrrMngt.AppEff)+100)/100;
            IrrReq = IrrReq*EffAdj;
            % Limit irrigation to maximum depth
            Irr = min(IrrMngt.MaxIrr,IrrReq);
        else
            % No irrigation
            Irr = 0;
        end
    elseif IrrMngt.IrrMethod == 3 % Irrigation - pre-defined schedule
        % Get current date
        CurrentDate = AOS_ClockStruct.StepStartTime;
        % Find irrigation value corresponding to current date
        Irr = IrrMngt.IrrigationSch((IrrMngt.IrrigationSch(:,1)==CurrentDate),2);
    elseif IrrMngt.IrrMethod == 4 % Irrigation - net irrigation
        % Net irrigation calculation performed after transpiration, so
        % irrigation is zero here
        Irr = 0;
    end
    % Update cumulative irrigation counter for growing season
    NewCond.IrrCum = NewCond.IrrCum+Irr;
elseif GrowingSeason == false
    % No irrigation outside growing season
    Irr = 0;
    NewCond.IrrCum = 0;
end

end

