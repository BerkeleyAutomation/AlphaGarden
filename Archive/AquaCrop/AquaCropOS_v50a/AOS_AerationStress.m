function [Ksa,NewCond] = AOS_AerationStress(Crop,InitCond,thRZ)
% Function to calculate aeration stress coefficient

%% Store initial conditions in new structure for updating %%
NewCond = InitCond;

%% Determine aeration stress (root zone) %%
if thRZ.Act > thRZ.Aer
    % Calculate aeration stress coefficient
    if NewCond.AerDays < Crop.LagAer
        stress = 1-((thRZ.Sat-thRZ.Act)/(thRZ.Sat-thRZ.Aer));
        Ksa.Aer = 1-((NewCond.AerDays/3)*stress);
    elseif NewCond.AerDays >= Crop.LagAer
        Ksa.Aer = (thRZ.Sat-thRZ.Act)/(thRZ.Sat-thRZ.Aer);
    end
    % Increment aeration days counter
    NewCond.AerDays = NewCond.AerDays+1;
    if NewCond.AerDays > Crop.LagAer
        NewCond.AerDays = Crop.LagAer;
    end
else
    % Set aeration stress coefficient to one (no stress value)
    Ksa.Aer = 1;
    % Reset aeration days counter
    NewCond.AerDays = 0;
end

end