function [NewCond] = AOS_GrowthStage(Crop,InitCond,GrowingSeason)
% Function to calculate number of growing degree days on current day

%% Store initial conditions in new structure for updating %% 
NewCond = InitCond;

%% Get growth stage (if in growing season) %%
if GrowingSeason == true
    % Adjust time for any delayed growth
    if Crop.CalendarType == 1
        tAdj = NewCond.DAP-NewCond.DelayedCDs;
    elseif Crop.CalendarType == 2
        tAdj = NewCond.GDDcum-NewCond.DelayedGDDs;
    end

    % Update growth stage
    if tAdj <= Crop.Canopy10Pct
        NewCond.GrowthStage = 1;
    elseif tAdj <= Crop.MaxCanopy
        NewCond.GrowthStage = 2;
    elseif tAdj <= Crop.Senescence
        NewCond.GrowthStage = 3;
    elseif tAdj > Crop.Senescence
        NewCond.GrowthStage = 4;
    end
else
    % Not in growing season so growth stage is set to dummy value 
    NewCond.GrowthStage = 0;
end

end

