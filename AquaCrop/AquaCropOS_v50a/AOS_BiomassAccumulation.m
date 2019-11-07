function [NewCond] = AOS_BiomassAccumulation(Crop,InitCond,Tr,TrPot,...
    Et0,Tmax,Tmin,GDD,GrowingSeason)
% Function to calculate biomass accumulation  

%% Store initial conditions in a new structure for updating %%
NewCond = InitCond;

%% Calculate biomass accumulation (if in growing season) %%
if GrowingSeason == true
    % Calculate temperature stress
    Kst = AOS_TemperatureStress(Crop,Tmax,Tmin,GDD);

    % Get time for harvest index build-up
    HIt = NewCond.DAP-NewCond.DelayedCDs-Crop.HIstartCD-1;

    if ((Crop.CropType == 2) || (Crop.CropType == 3)) && (NewCond.HIref > 0)
        % Adjust WP for reproductive stage
        if Crop.Determinant == 1
            fswitch = NewCond.PctLagPhase/100;
        else
            if HIt < (Crop.YldFormCD/3)
                fswitch = HIt/(Crop.YldFormCD/3);
            else
                fswitch = 1;
            end
        end
        WPadj = Crop.WP*(1-(1-Crop.WPy/100)*fswitch);
    else
        WPadj = Crop.WP;
    end

    % Adjust WP for CO2 effects
    WPadj = WPadj*Crop.fCO2;

    % Calculate biomass accumulation on current day
    % No water stress
    dB_NS = WPadj*(TrPot/Et0)*Kst.Bio;
    % With water stress
    dB = WPadj*(Tr/Et0)*Kst.Bio;
    if isnan(dB) == true
        dB = 0;
    end

    % Update biomass accumulation
    NewCond.B = NewCond.B+dB;
    NewCond.B_NS = NewCond.B_NS+dB_NS;
else
    % No biomass accumulation outside of growing season 
    NewCond.B = 0;
    NewCond.B_NS = 0;
end

end

