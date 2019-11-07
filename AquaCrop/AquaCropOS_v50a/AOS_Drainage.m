function [NewCond,DeepPerc,FluxOut] = AOS_Drainage(Soil,InitCond)
% Function to redistribute stored soil water

%% Store initial conditions in new structure for updating %%
NewCond = InitCond;

%% Preallocate arrays %%
thnew = zeros(1,Soil.nComp);
FluxOut = zeros(1,Soil.nComp);

%% Initialise counters and states %%
drainsum = 0;

%% Calculate drainage and updated water contents %%
for ii = 1:Soil.nComp
    % Specify layer for compartment
    layeri = Soil.Comp.Layer(ii);
    
    % Calculate drainage ability of compartment ii
    if InitCond.th(ii) <= InitCond.th_fc_Adj(ii)
        dthdt = 0;
    elseif InitCond.th(ii) >= Soil.Layer.th_s(layeri)
        dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
            Soil.Layer.th_fc(layeri));
        if (InitCond.th(ii)-dthdt) < InitCond.th_fc_Adj(ii)
            dthdt = InitCond.th(ii)-InitCond.th_fc_Adj(ii);
        end
    else      
        dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
            Soil.Layer.th_fc(layeri))*((exp(InitCond.th(ii)-...
            Soil.Layer.th_fc(layeri))-1)/(exp(Soil.Layer.th_s(layeri)-...
            Soil.Layer.th_fc(layeri))-1));
        if (InitCond.th(ii)-dthdt) < InitCond.th_fc_Adj(ii)
            dthdt = InitCond.th(ii)-InitCond.th_fc_Adj(ii);
        end
    end
    
    % Drainage from compartment ii (mm)
    draincomp = dthdt*Soil.Comp.dz(ii)*1000;
    
    % Check drainage ability of compartment ii against cumulative drainage
    % from compartments above
    excess = 0;
    prethick = Soil.Comp.dzsum(ii)-Soil.Comp.dz(ii);
    drainmax = dthdt*1000*prethick;
    if drainsum <= drainmax
        drainability = true;
    else
        drainability = false;
    end  
    
    % Drain compartment ii
    if drainability == true
        % No storage needed. Update water content in compartment ii
        thnew(ii) = InitCond.th(ii)-dthdt;

        % Update cumulative drainage (mm)
        drainsum = drainsum+draincomp;
        
        % Restrict cumulative drainage to saturated hydraulic
        % conductivity and adjust excess drainage flow
        if drainsum > Soil.Layer.Ksat(layeri)
            excess = excess+drainsum-Soil.Layer.Ksat(layeri);
            drainsum = Soil.Layer.Ksat(layeri);
        end
    elseif drainability == false
        % Storage is needed
        dthdt = drainsum/(1000*prethick);
        
        % Calculate value of theta (thX) needed to provide a
        % drainage ability equal to cumulative drainage
        if dthdt <= 0 
            thX = InitCond.th_fc_Adj(ii);
        elseif Soil.Layer.tau(layeri) > 0
            A = 1+((dthdt*(exp(Soil.Layer.th_s(layeri)-Soil.Layer.th_fc(layeri))-1))...
                /(Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-Soil.Layer.th_fc(layeri))));
            thX = Soil.Layer.th_fc(layeri)+log(A);
            if thX < InitCond.th_fc_Adj(ii)
                thX = InitCond.th_fc_Adj(ii);
            end
        else
            thX = Soil.Layer.th_s(layer)+0.01;
        end
        
        % Check thX against hydraulic properties of current soil layer
        if thX <= Soil.Layer.th_s(layeri)
            % Increase compartment ii water content with cumulative
            % drainage
            thnew(ii) = InitCond.th(ii)+(drainsum/(1000*Soil.Comp.dz(ii)));
            % Check updated water content against thX
            if thnew(ii) > thX
                % Cumulative drainage is the drainage difference
                % between theta_x and new theta plus drainage ability
                % at theta_x.
                drainsum = (thnew(ii)-thX)*1000*Soil.Comp.dz(ii);
                % Calculate drainage ability for thX
                if thX <= InitCond.th_fc_Adj(ii)
                    dthdt = 0;
                elseif thX >= Soil.Layer.th_s(layeri)
                    dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
                        Soil.Layer.th_fc(layeri));
                    if (thX-dthdt) < InitCond.th_fc_Adj(ii)
                        dthdt = thX-InitCond.th_fc_Adj(ii);
                    end
                else
                    dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
                        Soil.Layer.th_fc(layeri))*((exp(thX-Soil.Layer.th_fc(layeri))-1)...
                        /(exp(Soil.Layer.th_s(layeri)-Soil.Layer.th_fc(layeri))-1));
                    if (thX-dthdt) < InitCond.th_fc_Adj(ii)
                        dthdt = thX-InitCond.th_fc_Adj(ii);
                    end
                end
                % Update drainage total
                drainsum = drainsum+(dthdt*1000*Soil.Comp.dz(ii));
                % Restrict cumulative drainage to saturated hydraulic
                % conductivity and adjust excess drainage flow
                if drainsum > Soil.Layer.Ksat(layeri)
                    excess = excess+drainsum-Soil.Layer.Ksat(layeri);
                    drainsum = Soil.Layer.Ksat(layeri);
                end
                % Update water content
                thnew(ii) = thX-dthdt;
            elseif thnew(ii) > InitCond.th_fc_Adj(ii)
                % Calculate drainage ability for updated water content
                if thnew(ii) <= InitCond.th_fc_Adj(ii)
                    dthdt = 0;
                elseif thnew(ii) >= Soil.Layer.th_s(layeri)
                    dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
                        Soil.Layer.th_fc(layeri));
                    if (thnew(ii)-dthdt) < InitCond.th_fc_Adj(ii)
                        dthdt = thnew(ii)-InitCond.th_fc_Adj(ii);
                    end
                else
                    dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
                        Soil.Layer.th_fc(layeri))*((exp(thnew(ii)-...
                        Soil.Layer.th_fc(layeri))-1)/(exp(Soil.Layer.th_s(layeri)-...
                        Soil.Layer.th_fc(layeri))-1));
                    if (thnew(ii)-dthdt) < InitCond.th_fc_Adj(ii)
                        dthdt = thnew(ii)-InitCond.th_fc_Adj(ii);
                    end
                end
                % Update water content in compartment ii
                thnew(ii) = thnew(ii)-dthdt;
                % Update cumulative drainage
                drainsum = dthdt*1000*Soil.Comp.dz(ii);
                % Restrict cumulative drainage to saturated hydraulic
                % conductivity and adjust excess drainage flow
                if drainsum > Soil.Layer.Ksat(layeri)
                    excess = excess+drainsum-Soil.Layer.Ksat(layeri);
                    drainsum = Soil.Layer.Ksat(layeri);
                end
            else
                % Drainage and cumulative drainage are zero as water
                % content has not risen above field capacity in
                % compartment ii.
                drainsum = 0;
            end
        elseif thX > Soil.Layer.th_s(layeri)
            % Increase water content in compartment ii with cumulative
            % drainage from above
            thnew(ii) = InitCond.th(ii)+(drainsum/(1000*Soil.Comp.dz(ii)));
            % Check new water content against hydraulic properties of soil
            % layer
            if thnew(ii) <= Soil.Layer.th_s(layeri)
                if thnew(ii) > InitCond.th_fc_Adj(ii)
                    % Calculate new drainage ability
                    if thnew(ii) <= InitCond.th_fc_Adj(ii)
                        dthdt = 0;
                    elseif thnew(ii) >= Soil.Layer.th_s(layeri)
                        dthdt = Soil.Layer.tau(layeri)*...
                            (Soil.Layer.th_s(layeri)-Soil.Layer.th_fc(layeri));
                        if (thnew(ii)-dthdt) < InitCond.th_fc_Adj(ii)
                            dthdt = thnew(ii)-InitCond.th_fc_Adj(ii);
                        end
                    else
                        dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
                            Soil.Layer.th_fc(layeri))*((exp(thnew(ii)-...
                            Soil.Layer.th_fc(layeri))-1)/(exp(Soil.Layer.th_s(layeri)-...
                            Soil.Layer.th_fc(layeri))-1));
                        if (thnew(ii)-dthdt) < InitCond.th_fc_Adj(ii)
                            dthdt = thnew(ii)-InitCond.th_fc_Adj(ii);
                        end
                    end
                    % Update water content in compartment ii
                    thnew(ii) = thnew(ii)-dthdt;
                    % Update cumulative drainage
                    drainsum = dthdt*1000*Soil.Comp.dz(ii);
                    % Restrict cumulative drainage to saturated hydraulic
                    % conductivity and adjust excess drainage flow
                    if drainsum > Soil.Layer.Ksat(layeri)
                        excess = excess+drainsum-Soil.Layer.Ksat(layeri);
                        drainsum = Soil.Layer.Ksat(layeri);
                    end
                else
                    drainsum = 0;
                end
            elseif thnew(ii) > Soil.Layer.th_s(layeri)
                % Calculate excess drainage above saturation
                excess = (thnew(ii)-Soil.Layer.th_s(layeri))*1000*Soil.Comp.dz(ii);
                % Calculate drainage ability for updated water content
                if thnew(ii) <= InitCond.th_fc_Adj(ii)
                    dthdt = 0;
                elseif thnew(ii) >= Soil.Layer.th_s(layeri)
                    dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
                        Soil.Layer.th_fc(layeri));
                    if (thnew(ii)-dthdt) < InitCond.th_fc_Adj(ii)
                        dthdt = thnew(ii)-InitCond.th_fc_Adj(ii);
                    end
                else
                    dthdt = Soil.Layer.tau(layeri)*(Soil.Layer.th_s(layeri)-...
                        Soil.Layer.th_fc(layeri))*((exp(thnew(ii)-Soil.Layer.th_fc(layeri))-1)...
                        /(exp(Soil.Layer.th_s(layeri)-Soil.Layer.th_fc(layeri))-1));
                    if (thnew(ii)-dthdt) < InitCond.th_fc_Adj(ii)
                        dthdt = thnew(ii)-InitCond.th_fc_Adj(ii);
                    end
                end
                % Update water content in compartment ii
                thnew(ii) = Soil.Layer.th_s(layeri)-dthdt;
                
                % Update drainage from compartment ii
                draincomp = dthdt*1000*Soil.Comp.dz(ii);
                % Update maximum drainage
                drainmax = dthdt*1000*prethick;
                
                % Update excess drainage
                if drainmax > excess
                    drainmax = excess;
                end
                excess = excess-drainmax;
                % Update drainsum and restrict to saturated hydraulic
                % conductivity of soil layer
                drainsum = draincomp+drainmax;
                if drainsum > Soil.Layer.Ksat(layeri)
                    excess = excess+drainsum-Soil.Layer.Ksat(layeri);
                    drainsum = Soil.Layer.Ksat(layeri);
                end
            end
        end
    end
    
    % Store output flux from compartment ii
    FluxOut(ii) = drainsum;
    
    % Redistribute excess in compartment above
    if excess > 0
        precomp = ii+1;
        while (excess>0) && (precomp~=1)
            % Update compartment counter
            precomp = precomp-1;
            % Update layer counter
            layeri = Soil.Comp.Layer(precomp);
            % Update flux from compartment
            if precomp < ii
                FluxOut(precomp) = FluxOut(precomp)-excess;
            end
            % Increase water content to store excess
            thnew(precomp) = thnew(precomp)+(excess/(1000*Soil.Comp.dz(precomp)));
            
            % Limit water content to saturation and adjust excess counter
            if thnew(precomp) > Soil.Layer.th_s(layeri)
                excess = (thnew(precomp)-Soil.Layer.th_s(layeri))*1000*Soil.Comp.dz(precomp);
                thnew(precomp) = Soil.Layer.th_s(layeri);
            else
                excess = 0;
            end
        end
    end    
end

%% Update conditions and outputs %%
% Total deep percolation (mm)
DeepPerc = drainsum;
% Water contents
NewCond.th = thnew;

end

