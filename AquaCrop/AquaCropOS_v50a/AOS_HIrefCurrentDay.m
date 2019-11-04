function [NewCond] = AOS_HIrefCurrentDay(InitCond,Crop,GrowingSeason)
% Function to calculate reference (no adjustment for stress effects)
% harvest index on current day

%% Store initial conditions for updating %%
NewCond = InitCond;

%% Calculate reference harvest index (if in growing season) %%
if GrowingSeason == true
    % Check if in yield formation period
    if Crop.CalendarType == 1
        tAdj = NewCond.DAP-NewCond.DelayedCDs;
    elseif Crop.CalendarType == 2
        tAdj = NewCond.GDDcum-NewCond.DelayedGDDs;
    end
    if tAdj > Crop.HIstart
        NewCond.YieldForm = true;
    else
        NewCond.YieldForm = false;
    end 

    % Get time for harvest index calculation
    HIt = NewCond.DAP-NewCond.DelayedCDs-Crop.HIstartCD-1;

    if HIt <= 0
        % Yet to reach time for HI build-up
        NewCond.HIref = 0;
        NewCond.PctLagPhase = 0;
    else
        if NewCond.CCprev <= (Crop.CCmin*Crop.CCx)
            % HI cannot develop further as canopy cover is too small
            NewCond.HIref = InitCond.HIref;
        else
            % Check crop type
            if (Crop.CropType == 1) || (Crop.CropType == 2);
                % If crop type is leafy vegetable or root/tuber, then proceed with
                % logistic growth (i.e. no linear switch)
                NewCond.PctLagPhase = 100; % No lag phase
                % Calculate reference harvest index for current day
                NewCond.HIref = (Crop.HIini*Crop.HI0)/(Crop.HIini+...
                    (Crop.HI0-Crop.HIini)*exp(-Crop.HIGC*HIt));
                % Harvest index apprAOShing maximum limit
                if NewCond.HIref >= (0.9799*Crop.HI0)
                    NewCond.HIref = Crop.HI0;
                end
            elseif Crop.CropType == 3
                % If crop type is fruit/grain producing, check for linear switch
                if HIt < Crop.tLinSwitch
                    % Not yet reached linear switch point, therefore proceed with
                    % logistic build-up
                    NewCond.PctLagPhase = 100*(HIt/Crop.tLinSwitch);
                    % Calculate reference harvest index for current day
                    % (logistic build-up)
                    NewCond.HIref = (Crop.HIini*Crop.HI0)/(Crop.HIini+...
                        (Crop.HI0-Crop.HIini)*exp(-Crop.HIGC*HIt));
                else
                    % Linear switch point has been reached
                    NewCond.PctLagPhase = 100;        
                    % Calculate reference harvest index for current day
                    % (logistic portion)
                    NewCond.HIref = (Crop.HIini*Crop.HI0)/(Crop.HIini+...
                        (Crop.HI0-Crop.HIini)*exp(-Crop.HIGC*Crop.tLinSwitch));
                    % Calculate reference harvest index for current day
                    % (total - logistic portion + linear portion)
                    NewCond.HIref = NewCond.HIref+(Crop.dHILinear*...
                        (HIt-Crop.tLinSwitch));
                end

            end
            % Limit HIref and round off computed value
            if NewCond.HIref > Crop.HI0
                NewCond.HIref = Crop.HI0;
            elseif NewCond.HIref <= (Crop.HIini+0.004)
                NewCond.HIref = 0;
            elseif ((Crop.HI0-NewCond.HIref)<0.004)
                NewCond.HIref = Crop.HI0;
            end
        end
    end
else
    % Reference harvest index is zero outside of growing season
    NewCond.HIref = 0;
end

end

