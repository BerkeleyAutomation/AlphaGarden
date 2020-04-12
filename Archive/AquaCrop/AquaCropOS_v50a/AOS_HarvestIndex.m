function [ NewCond ] = AOS_HarvestIndex(Soil,Crop,InitCond,Et0,Tmax,...
    Tmin,GDD,GrowingSeason)
% Function to simulate build up of harvest index

%% Store initial conditions for updating %%
NewCond = InitCond;

%% Calculate harvest index build up (if in growing season) %%
if GrowingSeason == true
    % Calculate root zone water content
    [~,Dr,TAW,~] = AOS_RootZoneWater(Soil,Crop,NewCond);

    % Calculate water stress
    beta = true;
    Ksw = AOS_WaterStress(Crop,NewCond,Dr,TAW,Et0,beta);

    % Calculate temperature stress
    Kst = AOS_TemperatureStress(Crop,Tmax,Tmin,GDD);

    % Get reference harvest index on current day
    HIi = NewCond.HIref;

    % Get time for harvest index build-up
    HIt = NewCond.DAP-NewCond.DelayedCDs-Crop.HIstartCD-1;

    % Calculate harvest index
    if (NewCond.YieldForm == true) && (HIt >= 0)
        % Root/tuber or fruit/grain crops
        if (Crop.CropType == 2) || (Crop.CropType == 3)

            % Detemine adjustment for water stress before anthesis
            if InitCond.PreAdj == false
                NewCond.PreAdj = true;
                NewCond = AOS_HIadjPreAnthesis(NewCond,Crop);
            end

            % Determine adjustment for crop pollination failure
            if Crop.CropType == 3 % Adjustment only for fruit/grain crops
                if (HIt > 0) && (HIt <= Crop.FloweringCD)
                    NewCond = AOS_HIadjPollination(InitCond,Crop,...
                        Ksw,Kst,HIt);
                end
                HImax = NewCond.Fpol*Crop.HI0;
            else
                % No pollination adjustment for root/tuber crops
                HImax = Crop.HI0;
            end

            % Determine adjustments for post-anthesis water stress
            if HIt > 0 
                NewCond = AOS_HIadjPostAnthesis(NewCond,Crop,Ksw);
            end

            % Limit HI to maximum allowable increase due to pre- and
            % post-anthesis water stress combinations
            HImult = NewCond.Fpre*NewCond.Fpost;
            if HImult > 1+(Crop.dHI0/100)
                HImult = 1+(Crop.dHI0/100);
            end

            % Determine harvest index on current day, adjusted for stress
            % effects
            if HImax >= HIi
                HIadj = HImult*HIi;
            else
                HIadj = HImult*HImax;
            end
        elseif Crop.CropType == 1
            % Leafy vegetable crops - no adjustment, harvest index equal to
            % reference value for current day
            HIadj = HIi;
        end
    else
        % No build-up of harvest index if outside yield formation period
        HIi = InitCond.HI;
        HIadj = InitCond.HIadj;
    end
    % Store final values for current time step
    NewCond.HI = HIi;
    NewCond.HIadj = HIadj;
else
    % No harvestable crop outside of a growing season
    NewCond.HI = 0;
    NewCond.HIadj = 0;
end

end      