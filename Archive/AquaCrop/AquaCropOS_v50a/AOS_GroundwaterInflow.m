function [NewCond,GwIn] = AOS_GroundwaterInflow(Soil,InitCond)
% Function to calculate capillary rise in the presence of a shallow
% groundwater table

%% Store initial conditions for updating %%
NewCond = InitCond;
GwIn = 0;

%% Perform calculations %%
if NewCond.WTinSoil == true
    % Water table in soil profile. Calculate horizontal inflow.
    % Get groundwater table elevation on current day
    zGW = NewCond.zGW;
    
    % Find compartment mid-points
    zBot = cumsum(Soil.Comp.dz);
    zTop = zBot-Soil.Comp.dz;
    zMid = (zTop+zBot)/2;
    
    % For compartments below water table, set to saturation %
    idx = find(zMid >= zGW);
    for ii = idx:Soil.nComp
        % Get soil layer
        layeri = Soil.Comp.Layer(ii);
        if NewCond.th(ii) < Soil.Layer.th_s(layeri);
            % Update water content
            dth = Soil.Layer.th_s(layeri)-NewCond.th(ii);
            NewCond.th(ii) = Soil.Layer.th_s(layeri);
            % Update groundwater inflow
            GwIn = GwIn+(dth*1000*Soil.Comp.dz(ii));
        end
    end
end

end

