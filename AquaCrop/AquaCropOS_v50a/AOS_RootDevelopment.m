function [NewCond] = AOS_RootDevelopment(Crop,Soil,Groundwater,...
    InitCond,GDD,GrowingSeason)
% Function to calculate root zone expansion

%% Store initial conditions for updating %%
NewCond = InitCond;

%% Calculate root expansion (if in growing season) %%
if GrowingSeason == true
    % If today is first day of season, root depth is equal to minimum depth
    if NewCond.DAP == 1
        InitCond.Zroot = Crop.Zmin;
    end
    % Adjust time for any delayed development
    if Crop.CalendarType == 1
        tAdj = NewCond.DAP-NewCond.DelayedCDs;
    elseif Crop.CalendarType == 2
        tAdj = NewCond.GDDcum-NewCond.DelayedGDDs;
    end
    % Calculate root expansion %
    Zini = Crop.Zmin*(Crop.PctZmin/100);
    t0 = round((Crop.Emergence/2));
    tmax = Crop.MaxRooting;
    if Crop.CalendarType == 1
        tOld = tAdj-1;
    elseif Crop.CalendarType == 2
        tOld = tAdj-GDD;
    end

    % Potential root depth on previous day
    if tOld >= tmax
        ZrOld = Crop.Zmax;
    elseif tOld <= t0
        ZrOld = Zini;
    else
        X = (tOld-t0)/(tmax-t0);
        ZrOld = Zini+(Crop.Zmax-Zini)*nthroot(X,Crop.fshape_r);
    end
    if ZrOld < Crop.Zmin
        ZrOld = Crop.Zmin;
    end

    % Potential root depth on current day
    if tAdj >= tmax
        Zr = Crop.Zmax;
    elseif tAdj <= t0
        Zr = Zini;
    else
        X = (tAdj-t0)/(tmax-t0);
        Zr = Zini+(Crop.Zmax-Zini)*nthroot(X,Crop.fshape_r);
    end
    if Zr < Crop.Zmin
        Zr = Crop.Zmin;
    end
    % Determine rate of change
    dZr = Zr-ZrOld;
    % Adjust rate of expansion for any stomatal water stress
    if NewCond.TrRatio < 0.9999
        if Crop.fshape_ex >= 0
            dZr = dZr*NewCond.TrRatio;
        else
            fAdj = (exp(NewCond.TrRatio*Crop.fshape_ex)-1)/(exp(Crop.fshape_ex)-1);
            dZr = dZr*fAdj;
        end
    end
    % Adjust root expansion for failure to germinate (roots cannot expand
    % if crop has not germinated)
    if InitCond.Germination == false
        dZr = 0;
    end

    % Get new rooting depth
    NewCond.Zroot = InitCond.Zroot+dZr;

    % Adjust root depth if restrictive soil layer is present that limits 
    % depth of root expansion
    if Soil.zRes > 0
        if NewCond.Zroot > Soil.zRes
            NewCond.rCor = (2*(NewCond.Zroot/Soil.zRes)*((Crop.SxTop+Crop.SxBot)/2)...
                -Crop.SxTop)/Crop.SxBot;
            NewCond.Zroot = Soil.zRes;
        end
    end
    
    % Limit rooting depth if groundwater table is present (roots cannot
    % develop below the water table)
    if (Groundwater.WaterTable == 1) && (NewCond.zGW > 0)
        if NewCond.Zroot > NewCond.zGW
            NewCond.Zroot = NewCond.zGW;
            if NewCond.Zroot < Crop.Zmin
                NewCond.Zroot = Crop.Zmin;
            end
        end
    end    
else
    % No root system outside of the growing season
    NewCond.Zroot = 0;
end
end