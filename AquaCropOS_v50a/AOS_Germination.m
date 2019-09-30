function [NewCond] = AOS_Germination(InitCond,Soil,Crop,GDD,GrowingSeason)
% Function to check if crop has germinated

%% Store initial conditions in new structure for updating %%
NewCond = InitCond;

%% Check for germination (if in growing season) %%
if GrowingSeason == true
    % Find compartments covered by top soil layer affecting germination
    comp_sto = find(Soil.Comp.dzsum>=Soil.zGerm,1,'first');

    % Calculate water content in top soil layer
    Wr = 0;
    WrFC = 0;
    WrWP = 0;
    for ii = 1:comp_sto
        % Get soil layer
        layeri = Soil.Comp.Layer(ii);
        % Determine fraction of compartment covered by top soil layer
        if Soil.Comp.dzsum(ii) > Soil.zGerm
            factor = 1-((Soil.Comp.dzsum(ii)-Soil.zGerm)/Soil.Comp.dz(ii));
        else
            factor = 1;
        end
        % Increment actual water storage (mm)
        Wr = Wr+(factor*1000*InitCond.th(ii)*Soil.Comp.dz(ii));
        % Increment water storage at field capacity (mm)
        WrFC = WrFC+(factor*1000*Soil.Layer.th_fc(layeri)*Soil.Comp.dz(ii));
        % Increment water storage at permanent wilting point (mm)
        WrWP = WrWP+(factor*1000*Soil.Layer.th_wp(layeri)*Soil.Comp.dz(ii)); 
    end
    % Limit actual water storage to not be less than zero
    if Wr < 0
        Wr = 0;
    end
    % Calculate proportional water content
    WcProp = 1-((WrFC-Wr)/(WrFC-WrWP));

    % Check if water content is above germination threshold
    if (WcProp >= Crop.GermThr) && (NewCond.Germination == false)
        % Crop has germinated
        NewCond.Germination = true;
    end

    % Increment delayed growth time counters if germination is yet to occur
    if NewCond.Germination == false
        NewCond.DelayedCDs = InitCond.DelayedCDs+1;
        NewCond.DelayedGDDs = InitCond.DelayedGDDs+GDD;
    end
else
    % Not in growing season so no germination calculation is performed.
    NewCond.Germination = false;
    NewCond.DelayedCDs = 0;
    NewCond.DelayedGDDs = 0;
end

end

