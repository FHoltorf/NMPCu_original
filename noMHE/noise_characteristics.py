# relative standard deviation that is associated with disturbances
disturbance_error = 0.10
measurement_error = 0.10
disturbance_error_T = 0.0
disturbance_error_F = 0.0

qcov = {}
qcov[("PO",()), ("PO",())] = disturbance_error
qcov[("MX",(0,)), ("MX",(0,))] = disturbance_error
qcov[("MX",(1,)), ("MX",(1,))] = disturbance_error
qcov[("X",()), ("X",())] = disturbance_error
qcov[("MY",()), ("MY",())] = disturbance_error
qcov[("Y",()), ("Y",())] = disturbance_error
qcov[("W",()), ("W",())] = disturbance_error
qcov[("m_tot",()), ("m_tot",())] = 0.0
qcov[("PO_fed",()), ("PO_fed",())] = 0.0

# process disturbances
v_disturbances = {}
v_disturbances[("PO",())] = disturbance_error
v_disturbances[("MX",(0,))] = disturbance_error
v_disturbances[("MX",(1,))] = disturbance_error
v_disturbances[("X",())] = disturbance_error
v_disturbances[("MY",())] = disturbance_error
v_disturbances[("Y",())] = disturbance_error
v_disturbances[("W",())] = disturbance_error
v_disturbances[("m_tot",())] = disturbance_error*0.0
v_disturbances[("PO_fed",())] = disturbance_error*0.0
v_disturbances["u1"] = disturbance_error_T
v_disturbances["u2"] = disturbance_error_F

# relative variance (measurement noise) that is associated with the disturbances
mcov = {}
mcov[("PO",()), ("PO",())] = measurement_error
mcov[("MX",(0,)), ("MX",(0,))] = measurement_error
mcov[("MX",(1,)), ("MX",(1,))] = measurement_error
mcov[("X",()), ("X",())] = measurement_error
mcov[("MY",()), ("MY",())] = measurement_error
mcov[("Y",()), ("Y",())] = measurement_error
mcov[("W",()), ("W",())] = measurement_error
mcov[("MW",()), ("MW",())] = measurement_error
mcov[("m_tot",()), ("m_tot",())] = 0.005
mcov[("PO_fed",()), ("PO_fed",())] = 0.0 # measurement_error
mcov[(("heat_removal"),()),(("heat_removal"),())] =  measurement_error

# actual measurement noise from which measurement noise is generated
x_measurement = {}
x_measurement[("PO",())] = measurement_error
x_measurement[("MX",(0,))] = measurement_error
x_measurement[("MX",(1,))] = measurement_error
x_measurement[("X",())] = measurement_error
x_measurement[("MY",())] = measurement_error
x_measurement[("Y",())] = measurement_error
x_measurement[("W",())] = measurement_error
x_measurement[("MW",())] = measurement_error
x_measurement[("m_tot",())] = 0.005
x_measurement[("PO_fed",())] = 0.0
x_measurement[("heat_removal",())] =  measurement_error

# relative variance that is associated with the controls
ucov = {}
ucov[("u1",())] = disturbance_error_T
ucov[("u2",())] = disturbance_error_F

# uncertainty in initial point
v_init = {}
v_init[("PO",())] = 0.0
v_init[("MX",(0,))] = 0.0
v_init[("MX",(1,))] = 0.0
v_init[("X",())] = 0.0
v_init[("MY",())] = 0.0
v_init[("Y",())] = 0.0
v_init[("W",())] = 0.0
v_init[("m_tot",())] = 0.0
v_init[("PO_fed",())] = 0.0

# uncertainty in parameters
v_param = {} #[relative standard deviation, frequency for changes (measured in intervalls)]
v_param[('A',('a',))] = [0.0,200]
v_param[('A',('i',))] = [0.1,100]
v_param[('A',('p',))] = [0.1,100]
v_param[('A',('t',))] = [0.0,200]
v_param[('Ea',('a',))] = [0.0,200] 
v_param[('Ea',('i',))] = [0.0,200]  # have such strong impact that there is no way of makeing these uncertain
v_param[('Ea',('p',))] = [0.0,200]  # have such strong impact that there is no way of making these uncertain
v_param[('Ea',('t',))] = [0.0,200]
v_param[('Hrxn',('a',))] = [0.0,200] 
v_param[('Hrxn',('i',))] = [0.0,200]
v_param[('Hrxn',('p',))] = [0.0,200]
v_param[('Hrxn',('t',))] = [0.0,200]
v_param[('Hrxn_aux',('a',))] = [0.0,200] 
v_param[('Hrxn_aux',('i',))] = [0.0,200]
v_param[('Hrxn_aux',('p',))] = [0.0,200]
v_param[('Hrxn_aux',('t',))] = [0.0,200]
v_param[('kA',())] = [0.05,200]