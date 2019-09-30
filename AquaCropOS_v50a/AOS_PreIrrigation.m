function [NewCond,PreIrr] = AOS_PreIrrigation(Soil,Crop,IrrMngt,InitCond)
% Function to calculate pre-irrigation when in net irrigation mode

%% Store initial conditions for updating %%
NewCond = InitCond;

%% Calculate pre-irrigation needs %%
if (IrrMngt.IrrMethod ~= 4) || (NewCond.DAP ~= 1)
    % No pre-irrigation as not in net irrigation mode or not on first day
    % of the growing season
    PreIrr = 0;
else
    % Determine compartments covered by the root zone
    rootdepth = max(InitCond.Zroot,Crop.Zmin);
    rootdepth = round((rootdepth*100))/100;
    comp_sto = find(Soil.Comp.dzsum>=rootdepth,1,'first');
    % Calculate pre-irrigation requirements
    PreIrr = 0;
    for ii = 1:comp_sto
        % Get soil layer
        layeri = Soil.Comp.Layer(ii);
        % Determine critical water content threshold
        thCrit = Soil.Layer.th_wp(layeri)+((IrrMngt.NetIrrSMT/100)*...
            (Soil.Layer.th_fc(layeri)-Soil.Layer.th_wp(layeri)));
        % Check if pre-irrigation is required
        if NewCond.th(ii) < thCrit
            PreIrr = PreIrr+((thCrit-NewCond.th(ii))*1000*Soil.Comp.dz(ii));
            NewCond.th(ii) = thCrit;
        end
    end
end

end

