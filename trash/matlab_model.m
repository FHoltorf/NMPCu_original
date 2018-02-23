function dxdt = matlab_model(t,x,u1,u2)
    W_scale = 1.0;
    Y_scale = 1.0e-2;
    PO_scale = 1.0e1;
    MY0_scale = 1.0e-1;
    MX0_scale = 1.0e1;
    MX1_scale = 1.0e2;
    m_tot_scale = 1.0e4;
    T_scale = 1.0e2;
    % ODE System
    n_KOH = 2.70005346641; % (kmol)
    mw_PO = 58.08; % (kg/kmol)
    bulk_cp_1 = 1.1; % (kJ/kg/K)
    bulk_cp_2 = 2.72e-3;
    mono_cp_1 = 0.92; % (kJ/kg/K)
    mono_cp_2 = 8.87e-3;
    mono_cp_3 = -3.10e-5;
    mono_cp_4 = 4.78e-8;
    Hrxn = 92048.0; % (kJ/kmol)
    Hrxn_aux = 1.0;
    A_a = 8.64e4*60.0*1000.0; % (m^3/min/kmol)
    A_i = 3.964e5*60.0*1000.0;
    A_p = 1.35042e4*60.0*1000.0;
    A_t = 1.509e6*60.0*1000.0;
    Ea_a = 82.425; % (kJ/kmol)
    Ea_i = 77.822;
    Ea_p = 69.172;
    Ea_t = 105.018;
    kA = 2200.0*60.0/30.0/Hrxn; % (kJ/min/K)
    Tf = 298.15;

    dxdt = zeros(10,1);
    PO = x(1)*PO_scale;
    W = x(2)*W_scale;
    Y = x(3)*Y_scale;
    m_tot = x(4)*m_tot_scale;
    MX0 = x(5)*MX0_scale;
    MY = x(7)*MY0_scale;
    T = x(8)*T_scale;
    T_cw = x(9)*T_scale ;      
    %F = x(10) %pwa
    F = u2; %pwc
    kr_i = A_i * exp(-Ea_i/(8.314e-3*T)); % (m^3/min/kmol) * exp([kJ/kmol)/[kJ/kmol/K]/[K]) = [m^3/min/kmol]
    kr_p = A_p * exp(-Ea_p/(8.314e-3*T));
    kr_a = A_a * exp(-Ea_a/(8.314e-3*T));
    kr_t = A_t * exp(-Ea_t/(8.314e-3*T));
    X = 10.0432852386*2 + 47.7348816877 - 2*W - MX0; % [kmol]
    G = X*n_KOH/(X+MX0+Y+MY); % [kmol]
    U = Y*n_KOH/(X+MX0+Y+MY); % [kmol]
    MG = MX0*n_KOH/(X+MX0+Y+MY); % [kmol]

    monomer_cooling = (mono_cp_1*(T - Tf) + mono_cp_2/2.0 *(T^2.0-Tf^2.0) + mono_cp_3/3.0*(T^3.0-Tf^3.0) + mono_cp_4/4.0*(T^4.0-Tf^4.0))/Hrxn; %
    Vi = 1.0e3/m_tot/(1.0+0.0007576*(T - 298.15)); % 1/[m^3]
    Qr = (((kr_i-kr_p)*(G+U) + (kr_p + kr_t)*n_KOH + kr_a*W)*PO*Vi*Hrxn_aux - kr_a * W * PO * Vi * Hrxn_aux); % [m^3/min/kmol]*[kmol] *[kmol] *[1/m^3] * [kJ/kmol] = [kJ/min]
    Qc = kA * (T - T_cw); % [kJ/min/K] *[K] = [kJ/min]

    % PO
    dxdt(1) = (F - ((kr_i-kr_p)*(G+U)+(kr_p+kr_t)*n_KOH+kr_a*W) * PO * Vi)/PO_scale; % [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
    % W
    dxdt(2) = -kr_a * W * PO * Vi/W_scale; % [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
    % Y
    dxdt(3) = (kr_t*n_KOH-kr_i*U)*PO*Vi/Y_scale; % [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
    % m_tot
    dxdt(4) = mw_PO*F/m_tot_scale; % [kg/kmol] * [kmol/min] = [kg/min]
    % MX0
    dxdt(5) = kr_i*G*PO*Vi/MX0_scale; % [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
    % MX1
    dxdt(6) = (kr_i*G+kr_p*MG)*PO*Vi/MX1_scale; % [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
    % MY  
    dxdt(7) = kr_i*U*PO*Vi/MY0_scale;
    % T
    dxdt(8) = (Qr - Qc - F * monomer_cooling*mw_PO)*Hrxn/(m_tot*(bulk_cp_1 + bulk_cp_2*T))/T_scale; % [kJ/min] / [kg] / [kJ/kg/K] = [K/min]
    % T_cw
    dxdt(9) = u1/T_scale;
    % t
    dxdt(10) = 1.0;
    % pwa F