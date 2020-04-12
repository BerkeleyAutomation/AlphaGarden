function [Kst] = AOS_TemperatureStress(Crop,Tmax,Tmin,GDD)
% Function to calculate temperature stress coefficients

%% Calculate temperature stress coefficient affecting biomass growth %%
% Get parameters for logistic curve
KsBio_up = 1;
KsBio_lo = 0.02;
fshapeb = (-1)*(log(((KsBio_lo*KsBio_up)-0.98*KsBio_lo)...
    /(0.98*(KsBio_up-KsBio_lo))));

% Calculate temperature stress effects on biomass production
if Crop.BioTempStress == 0
    % No temperature stress effects on biomass production
    Kst.Bio = 1;
elseif Crop.BioTempStress == 1
    % Biomass production affected by temperature stress
    if GDD >= Crop.GDD_up
        Kst.Bio = 1;
    elseif GDD <= Crop.GDD_lo
        Kst.Bio = 0;
    else
        GDDrel = (GDD-Crop.GDD_lo)/(Crop.GDD_up-Crop.GDD_lo);
        Kst.Bio = (KsBio_up*KsBio_lo)/(KsBio_lo+(KsBio_up-KsBio_lo)...
            *exp(-fshapeb*GDDrel));
        Kst.Bio = Kst.Bio-KsBio_lo*(1-GDDrel);
    end
end

%% Calculate temperature stress coefficients affecting crop pollination %%
% Get parameters for logistic curve
KsPol_up = 1;
KsPol_lo = 0.001;

% Calculate effects of heat stress on pollination
if Crop.PolHeatStress == 0
    % No heat stress effects on pollination
    Kst.PolH = 1;
elseif Crop.PolHeatStress == 1
    % Pollination affected by heat stress
    if Tmax <= Crop.Tmax_lo
        Kst.PolH = 1;
    elseif Tmax >= Crop.Tmax_up
        Kst.PolH = 0;
    else
        Trel = (Tmax-Crop.Tmax_lo)/(Crop.Tmax_up-Crop.Tmax_lo);
        Kst.PolH = (KsPol_up*KsPol_lo)/(KsPol_lo+(KsPol_up-KsPol_lo)...
            *exp(-Crop.fshape_b*(1-Trel)));
    end
end

% Calculate effects of cold stress on pollination
if Crop.PolColdStress == 0
    % No cold stress effects on pollination
    Kst.PolC = 1;
elseif Crop.PolColdStress == 1
    % Pollination affected by cold stress
    if Tmin >= Crop.Tmin_up
        Kst.PolC = 1;
    elseif Tmin <= Crop.Tmin_lo
        Kst.PolC = 0;
    else
        Trel = (Crop.Tmin_up-Tmin)/(Crop.Tmin_up-Crop.Tmin_lo);
        Kst.PolC = (KsPol_up*KsPol_lo)/(KsPol_lo+(KsPol_up-KsPol_lo)...
            *exp(-Crop.fshape_b*(1-Trel)));
    end
end

end

