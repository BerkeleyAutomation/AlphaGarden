function [Wr,Dr,TAW,thRZ] = AOS_RootZoneWater(Soil,Crop,InitCond)
% Function to calculate actual and total available water in the root
% zone at current time step

%% Calculate root zone water content and available water %%
% Compartments covered by the root zone
rootdepth = max(InitCond.Zroot,Crop.Zmin);
rootdepth = round((rootdepth*100))/100;
comp_sto = sum(Soil.Comp.dzsum<rootdepth)+1;
% Initialise counters
Wr = 0;
WrS = 0;
WrFC = 0;
WrWP = 0;
WrDry = 0;
WrAer = 0;

for ii = 1:comp_sto
    % Specify layer
    layeri = Soil.Comp.Layer(ii);
    % Fraction of compartment covered by root zone
    if Soil.Comp.dzsum(ii) > rootdepth
        factor = 1-((Soil.Comp.dzsum(ii)-rootdepth)/Soil.Comp.dz(ii));
    else
        factor = 1;
    end
    % Actual water storage in root zone (mm)
    Wr = Wr+(factor*1000*InitCond.th(ii)*Soil.Comp.dz(ii));
    % Water storage in root zone at saturation (mm)
    WrS = WrS+(factor*1000*Soil.Layer.th_s(layeri)*Soil.Comp.dz(ii));
    % Water storage in root zone at field capacity (mm)
    WrFC = WrFC+(factor*1000*Soil.Layer.th_fc(layeri)*Soil.Comp.dz(ii));
    % Water storage in root zone at permanent wilting point (mm)
    WrWP = WrWP+(factor*1000*Soil.Layer.th_wp(layeri)*Soil.Comp.dz(ii)); 
    % Water storage in root zone at air dry (mm)
    WrDry = WrDry+(factor*1000*Soil.Layer.th_dry(layeri)*Soil.Comp.dz(ii));
    % Water storage in root zone at aeration stress threshold (mm)
    WrAer = WrAer+(factor*1000*(Soil.Layer.th_s(layeri)-(Crop.Aer/100))*Soil.Comp.dz(ii));
end

if Wr < 0
    Wr = 0;
end

% Actual root zone water content (m3/m3)
thRZ.Act = Wr/(rootdepth*1000);
% Root zone water content at saturation (m3/m3)
thRZ.Sat = WrS/(rootdepth*1000);
% Root zone water content at field capacity (m3/m3)
thRZ.Fc = WrFC/(rootdepth*1000);
% Root zone water content at permanent wilting point (m3/m3)
thRZ.Wp = WrWP/(rootdepth*1000);
% Root zone water content at air dry (m3/m3)
thRZ.Dry = WrDry/(rootdepth*1000);
% Root zone water content at aeration stress threshold (m3/m3)
thRZ.Aer = WrAer/(rootdepth*1000);

%% Calculate total available water (mm) %%
TAW = WrFC-WrWP;
if TAW < 0
    TAW = 0;
end

%% Calculate depletion (mm) %%
Dr = WrFC-Wr;
if Dr < 0
    Dr = 0;
end

end

