function [NewCond] = AOS_CheckGroundwaterTable(Soil,Groundwater,InitCond)
% Function to check for presence of a groundwater table, and, if present,
% to adjust compartment water contents and field capacities where
% necessary

%% Store initial conditions for updating %%
NewCond = InitCond;

%% Perform calculations (if variable water table is present) %%
if (Groundwater.WaterTable == 1) && (strcmp(Groundwater.Method,'Variable'))
    % Update groundwater conditions for current day
    NewCond.zGW = Groundwater.zGW((Groundwater.zGW(:,1)==ClockStruct.StepStartTime),2);

    % Find compartment mid-points
    zBot = cumsum(Soil.Comp.dz);
    zTop = zBot-Soil.Comp.dz;
    zMid = (zTop+zBot)/2;

    % Check if water table is within modelled soil profile
    if NewCond.zGW >= 0
        if isempty(find(zMid>=NewCond.zGW,1))
            NewCond.WTinSoil = false;
        else
            NewCond.WTinSoil = true;
        end
    end

    % If water table is in soil profile, adjust water contents
    if NewCond.WTinSoil == true
        idx = find(zMid >= NewCond.zGW);
        for ii = idx:Soil.nComp
            layeri = Soil.Comp.Layer(ii);
            NewCond.th(ii) = Soil.Layer.th_s(layeri);
        end
    end

    % Adjust compartment field capacity
    compi = Soil.nComp;
    thfcAdj = zeros(1,compi);
    % Find thFCadj for all compartments
    while compi >= 1
        layeri = Soil.Comp.Layer(compi);
        if Soil.Layer.th_fc(layeri) <= 0.1
            Xmax = 1;
        else
            if Soil.Layer.th_fc(layeri) >= 0.3
                Xmax = 2;
            else
                pF = 2+0.3*(Soil.Layer.th_fc(layeri)-0.1)/0.2;
                Xmax = (exp(pF*log(10)))/100;
            end
        end
        if (NewCond.zGW < 0) || ((NewCond.zGW-zMid(compi)) >= Xmax)
            for ii = 1:compi
                layerii = Soil.Comp.Layer(ii);
                thfcAdj(ii) = Soil.Layer.th_fc(layerii);
            end
            compi = 0;
        else
            if Soil.Layer.th_fc(layeri) >= Soil.Layer.th_s(layeri)
                thfcAdj(compi) = Soil.Layer.th_fc(layeri);
            else
                if zMid(compi) >= NewCond.zGW
                    thfcAdj(compi) = Soil.Layer.th_s(layeri);
                else
                    dV = Soil.Layer.th_s(layeri)-Soil.Layer.th_fc(layeri);
                    dFC = (dV/(Xmax^2))*((zMid(compi)-(NewCond.zGW-Xmax))^2);
                    thfcAdj(compi) = Soil.Layer.th_fc(layeri)+dFC;
                end
            end
            compi = compi-1;
        end
    end
    % Store adjusted field capacity values
    NewCond.th_fc_Adj = thfcAdj; 
end

end

