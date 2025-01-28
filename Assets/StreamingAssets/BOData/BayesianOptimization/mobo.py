from matplotlib import pyplot as plt
from import_all import *
import socket
import pickle
import pandas as pd
import time
import platform
import csv

# Define default values for global variables
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

CURRENT_DIR = ""
PROJECT_PATH = ""
OBSERVATIONS_LOG_PATH = ""

N_INITIAL = 5 # Number of sampling iterations
N_ITERATIONS = 10  # Number of optimization iterations

BATCH_SIZE = 1  # Number of design parameter points to query at the next iteration
NUM_RESTARTS = 10  # Used for the acquisition function number of restarts in optimization
RAW_SAMPLES = 1024  # Durch höhere RawSamples kein OptimierungsFehler (Optimization failed within `scipy.optimize.minimize` with status 1.')
MC_SAMPLES = 512  # Number of samples to approximate acquisition function
SEED = 3  # Seed to initialize the initial samples obtained

PROBLEM_DIM = 16 #dimension of the parameters x
NUM_OBJS = 2 #dimension of the objectives y

WARM_START = False #true if there is initial data (accessible from the following paths) that should be used before optimization restarts
CSV_PATH_PARAMETERS = ""
CSV_PATH_OBJECTIVES = ""

USER_ID = ""
CONDITION_ID = ""
GROUP_ID = ""

### Init socket and start receiving data
HOST = ''
PORT = 56001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print('Server starts, waiting for connection...', flush=True)
conn, addr = s.accept()
print('Connected by', addr, flush=True)

# wait for data ...
data = conn.recv(1024)

### Init hyperparameters received from Unity
init_data = data.decode("utf-8").split('_')
# Parse and update global variables
param_list = init_data[0].split(',')
BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, N_ITERATIONS, MC_SAMPLES, N_INITIAL, SEED, PROBLEM_DIM, NUM_OBJS = map(int, param_list)
warm_start_settings = init_data[1].split(',')
WARM_START = warm_start_settings[0].lower() in ("true", "1", "yes")
CSV_PATH_PARAMETERS = str(warm_start_settings[1])
CSV_PATH_OBJECTIVES = str(warm_start_settings[2])
print("Initialization parameters received and set.", flush=True)
print(f"BATCH_SIZE: {BATCH_SIZE}, NUM_RESTARTS: {NUM_RESTARTS}, RAW_SAMPLES: {RAW_SAMPLES}, N_ITERATIONS: {N_ITERATIONS}, MC_SAMPLES: {MC_SAMPLES}, N_INITIAL: {N_INITIAL}, SEED: {SEED}, PROBLEM_DIM: {PROBLEM_DIM}, NUM_OBJS: {NUM_OBJS}", flush=True)

### iter, init_sample, design_parameter_num, objective_num
parameter_raw = init_data[2].split('/')
print('Parameter', parameter_raw, flush=True)
parameters_strinfo = []
parameters_info = []
for i in range(len(parameter_raw) ):
    parameters_strinfo.append(parameter_raw[i].split(','))
for strlist in parameters_strinfo:
    parameters_info.append(list(map(float, strlist)))

objective_raw = init_data[3].split('/')
print('Objective', objective_raw, flush=True)
objectives_strinfo = []
objectives_info = []
for i in range(len(objective_raw)):
    objectives_strinfo.append(objective_raw[i].split(','))
for strlist in objectives_strinfo:
    objectives_info.append(list(map(float, strlist)))

print("Objectives info", len(objectives_info), flush=True)

### Init the parameter and objective names
parameter_names_raw = init_data[4].split(',')
parameter_names = []
for param_name in parameter_names_raw:
    parameter_names.append(param_name)

objective_names_raw = init_data[5].split(',')
objective_names = []
for obj_name in objective_names_raw:
    objective_names.append(obj_name)

### Init user information
study_info_raw = init_data[6].split(',')
USER_ID = study_info_raw[0]
CONDITION_ID = study_info_raw[1]
GROUP_ID = study_info_raw[2]

### Init the device and other settings
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference point in objective function space
# ... this tells the optimizer that all objectives should be maximized, having a -1 to 1 range, where higher is better
#ref_point = torch.tensor([-1. for _ in range(num_objs)]).cuda()
ref_point = torch.tensor([-1. for _ in range(NUM_OBJS)]).to(device)
#print("Ref_point", ref_point)

# Design parameter bounds to range from 0 to 1
problem_bounds = torch.zeros(2, PROBLEM_DIM, **tkwargs)
problem_bounds[1] = 1  # Set upper bounds
#print("Problem_bounds: ", problem_bounds, flush=True)

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

# -------------------------------------------------------
# Sample the current objective function from the Unity application
# -------------------------------------------------------
#  do this in EACH iteration of the defined total optimization iterations (= n_samples + n_iterations)
def objective_function(x_tensor):
    x = x_tensor.cpu().numpy()
    #print("x", x, flush=True)
    #print("Parameters_info:", parameters_info, flush=True)
    send_data = "parameters,"
    
    # Denormalize x_numpy using the provided function
    x_denormalized = np.array([denormalize_to_original_param(x[i], parameters_info[i][0], parameters_info[i][1]) for i in range(len(x))])
    
    # Prepare parameter string for sending to Unity
    for i in range(len(x_denormalized)):
        send_data += str(round(x_denormalized[i], 3)) + ","
        
    # Remove trailing comma
    send_data = send_data[:-1]
    
    # Send parameter data to Unity
    print("Send Data: ", send_data, flush=True)
    conn.sendall(bytes(send_data, 'utf-8'))

    # ......
    # ...... Wait for Unity to respond ......
    # ......
    
    # received data from Unity
    data = conn.recv(1024)
    received_objective = []
    if data:
        received_objective = list(map(float, data.decode("utf-8").split(',')))

    if len(data) == 0:
        print("unity end", flush=True)

    if len(received_objective) != NUM_OBJS:
        print("received objective number not consistent", flush=True)

    print("Received Objective Values: ", received_objective, flush=True)

    def limit_range(f):
        if f > 1:
            f = 1
        elif f < -1:
            f = -1
        return f

    #print("Received Number of Objectives", len(received_objective), flush=True)

    # Normalization
    fs = []
    for i in range(NUM_OBJS):
        # ... normalize values within the range -1 to 1:
        f = (received_objective[i] - objectives_info[i][0]) / (objectives_info[i][1] - objectives_info[i][0])
        f = f * 2 - 1
        if objectives_info[i][2] == 1: # ... invert if the objective was set to "smaller is better" (i.e., minimization)
            f *= -1
        f = limit_range(f)    
        fs.append(f)

    print("Objective normalized:", fs)
    return torch.tensor(fs, dtype=torch.float64).to(device)
    # return torch.tensor(fs, dtype=torch.float64).cuda()


# Parameter values range from 0 to 1
def denormalize_to_original_param(value, lower_bound, upper_bound):
    result = lower_bound + value * (upper_bound - lower_bound)
    return np.round(result, 2) if isinstance(result, (float, int)) else np.round(result, 2)

# Objective values range from -1 to 1
def denormalize_to_original_obj(value, lower_bound, upper_bound, smaller_is_better):
    if smaller_is_better == 1:  # ... if the objective was set to minimize, invert it first before denormalizing
        value *= -1
    result = lower_bound + (value + 1) / 2 * (upper_bound - lower_bound)
    return np.round(result, 2) if isinstance(result, (float, int)) else np.round(result, 2)


# -------------------------------------------------------
# create_csv_file
# -------------------------------------------------------
def create_csv_file(csv_file_path, fieldnames):
    try:
        if not os.path.exists(os.path.dirname(csv_file_path)):
            os.makedirs(os.path.dirname(csv_file_path))

        write_header = not os.path.exists(csv_file_path)

        with open(csv_file_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            if write_header:
                writer.writeheader()
    except Exception as e:
        print("Error creating file:", str(e), flush=True)

# -------------------------------------------------------
# write_data_to_csv
# -------------------------------------------------------
def write_data_to_csv(csv_file_path, fieldnames, data):
    try:
        with open(csv_file_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writerows(data)
    except Exception as e:
        print("Error writing to file:", str(e), flush=True)

# -------------------------------------------------------
# generate_initial_data
# -------------------------------------------------------
# This means that the optimization function always starts randomly and must be directly connected to the application.
# n_samples must be 2(d+1), where d = problem_dim (i.e., the number of design parameters) 
# (https://botorch.org/tutorials/multi_objective_bo)
def generate_initial_data(n_samples):
    
    # Define the CSV file path
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData", USER_ID)
    if not os.path.exists(PROJECT_PATH):
        os.makedirs(PROJECT_PATH)
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")

    # Check if the file exists and write the header if it does not
    if not os.path.exists(OBSERVATIONS_LOG_PATH):
        header = np.array(['UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase', 'IsPareto'] + objective_names + parameter_names)
        with open(OBSERVATIONS_LOG_PATH, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(header)

    # Generate training data
    train_x = draw_sobol_samples(
        bounds=problem_bounds, n=1, q=n_samples, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0)
    print("Initial training data (Sobol samples) in normalized range [-1, 1]:", train_x, flush=True)

    # loop to sample objective values from the users to these training data
    train_obj = []
    for i, x in enumerate(train_x):
        print(f"----------------------Initial Sample: {i + 1}", flush=True)
        obj = objective_function(x)
        train_obj.append(obj)

        # Convert objective and parameter values to numpy arrays for logging
        x_numpy = x.cpu().numpy()
        obj_numpy = obj.cpu().detach().numpy()
        # Denormalize x_numpy and obj_numpy
        x_denormalized = np.array([denormalize_to_original_param(x_numpy[i], parameters_info[i][0], parameters_info[i][1]) for i in range(len(x_numpy))])
        obj_denormalized = np.array([denormalize_to_original_obj(obj_numpy[i], objectives_info[i][0], objectives_info[i][1], objectives_info[i][2]) for i in range(len(obj_numpy))])
        # Combine all records and pareto identification
        all_record = np.concatenate(([USER_ID], [CONDITION_ID], [GROUP_ID], [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())], [i+1], ['sampling'], ['FALSE'], obj_denormalized, x_denormalized))
        # Log the values to the CSV file
        with open(OBSERVATIONS_LOG_PATH, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(all_record)

    train_obj_array = np.array([item.cpu().detach().numpy() for item in train_obj], dtype=np.float64)
    #print("Shape der Arrays: ", train_x.shape, torch.tensor(train_obj_array).to(device).shape)

    return train_x, torch.tensor(train_obj_array).to(device)

# -------------------------------------------------------
# load_data if there are prior observations
# -------------------------------------------------------
# (each row in the CSV represents one sampling iteration)
def load_data():
    # Define the CSV file path
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    objective_path = os.path.join(CURRENT_DIR, "InitData", CSV_PATH_OBJECTIVES)
    parameter_path = os.path.join(CURRENT_DIR, "InitData", CSV_PATH_PARAMETERS)
    # load the data from the provided file paths
    data_objectives = pd.read_csv(objective_path,  delimiter=';')
    data_parameter = pd.read_csv(parameter_path,  delimiter=';')

    y = torch.tensor(data_objectives.values, dtype=torch.float64).to(device)
    x = torch.tensor(data_parameter.values, dtype=torch.float64).to(device)

    return x, y

# -------------------------------------------------------
# initialize_model
# -------------------------------------------------------
def initialize_model(train_x, train_obj):
    # Normalize train_x to be between 0 and 1
    x_min = train_x.min(dim=0)[0]
    x_max = train_x.max(dim=0)[0]
    
    # Avoid division by zero in case of a constant parameter
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    
    train_x_normalized = (train_x - x_min) / x_range
    #print(f"Normalized train parameters. Min: {x_min}, Max: {x_max}", flush=True)  # Log normalization

    # Standardize objectives manually using the entire dataset
    train_obj_mean = train_obj.mean(dim=0)
    train_obj_std = train_obj.std(dim=0)
    
    # Avoid division by zero in case of zero standard deviation
    train_obj_std[train_obj_std == 0] = 1.0
    
    train_obj_standardized = (train_obj - train_obj_mean) / train_obj_std
    #print(f"Standardized train objectives. Mean: {train_obj_mean}, Std: {train_obj_std}", flush=True)  # Log standardization

    # Define models for objective and constraint
    #print("Initializing SingleTaskGP model...", flush=True)  # Log model initialization
    model = SingleTaskGP(train_x_normalized, train_obj_standardized)  # Use normalized train_x and standardized train_obj
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Log the state of the model and its parameters
    #print(f"Model initialized. Train X shape: {train_x.shape}, Train Obj shape: {train_obj.shape}", flush=True)
    #print(f"Model parameters: {list(model.parameters())}", flush=True)

    return mll, model

# -------------------------------------------------------
# optimize_qehvi
# -------------------------------------------------------
def optimize_qehvi(model, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    #print("Partitioning non-dominated space for acquisition function...", flush=True)  # Log partitioning
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj)
    
    #print("Setting up qEHVI acquisition function...", flush=True)  # Log acquisition function setup
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
    )
    
    # Log the current state of the acquisition function
    #print("Acquisition function set up successfully.", flush=True)

    # optimize
    #print("Optimizing acquisition function to find new candidates...", flush=True)  # Log optimization
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=problem_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for initialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    
    # Add small random noise to encourage exploration
    #noise = torch.randn_like(candidates) * 0.01
    #candidates = candidates + noise
    
    # Log the new candidate solutions found
    #print(f"Candidate solutions found: {candidates}", flush=True)

    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    #print(f"New candidate solution after unnormalizing: {new_x}", flush=True)  # Log unnormalized candidates
    return new_x

# -------------------------------------------------------
# mobo_execute ... this is the main method that handles the sampling and optimization loops
# -------------------------------------------------------
def mobo_execute(seed, iterations, initial_samples):
    #-----------------------
    # prepare file logging
    #-----------------------
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData", USER_ID)
    os.makedirs(PROJECT_PATH, exist_ok=True)
    csv_file_path = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(csv_file_path, ['Optimization', 'Execution_Time'])
    #-----------------------

    #-----------------------
    # init mobo
    #-----------------------
    torch.manual_seed(seed)
    #print(f"Seed set for reproducibility: {seed}", flush=True)  # Logging seed setting

    hv = Hypervolume(ref_point=ref_point)
    # Hypervolumes
    hvs_qehvi = []

    # Whether the optimizer explore first with initial data (warm-start) or random points in the parameter space...
    if WARM_START:
        print("Loading warm start data...", flush=True)  # Logging warm start
        train_x_qehvi, train_obj_qehvi = load_data()
    else:
        print("Generating initial training data...", flush=True)  # Logging data generation
        train_x_qehvi, train_obj_qehvi = generate_initial_data(n_samples=initial_samples)

    # Initialize GP models
    #print("Initializing GP models...", flush=True)  # Logging model initialization
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    print(f"Initial train_x_qehvi shape: {train_x_qehvi.shape}, train_obj_qehvi shape: {train_obj_qehvi.shape}", flush=True)  # Logging shapes

    # Check if initial data is reasonable
    if torch.any(train_obj_qehvi > 1) or torch.any(train_obj_qehvi < -1):
        print("Warning: Initial objective values are out of expected range [-1, 1].", flush=True)

    # Compute and save hypervolume for the sampling phase before optimization
    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]
    volume = hv.compute(pareto_y)
    hvs_qehvi.append(volume)
    #print(f"Initial hypervolume: {volume}", flush=True)  # Logging initial hypervolume
    save_hypervolume_to_file(hvs_qehvi, 0)
    #-----------------------

    #-----------------------
    # Go through the iterations
    #-----------------------
    for iteration in range(1, iterations + 1):
        print(f"----------------------MOBO Iteration: {iteration}", flush=True)
        # Startzeitpunkt der Iteration von mobo
        start_time = time.time()
        # Fit Models
        #print("Fitting GP model...", flush=True)  # Logging model fitting
        fit_gpytorch_mll(mll_qehvi)

        # Check if model fitting produced any NaN values
        if torch.any(torch.isnan(list(model_qehvi.parameters())[0].clone().detach())):
            print("Warning: NaN detected in GP model parameters after fitting.", flush=True)

        # Define qEI acquisition modules using the QMC sampler
        sample_shape = torch.Size([MC_SAMPLES])
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=sample_shape, seed=SEED)
        #print(f"Defined qEI acquisition module with sample shape: {sample_shape}", flush=True)  # Logging acquisition setup

        # Optimize acquisition functions and get new observations
        #print("Optimizing acquisition function...", flush=True)  # Logging acquisition optimization
        new_x_qehvi = optimize_qehvi(model_qehvi, train_obj_qehvi, qehvi_sampler)
        #print(f"New candidate solution: {new_x_qehvi}", flush=True)  # Logging new candidate solution

        # Check if the candidate solution is within expected bounds
        if torch.any(new_x_qehvi < problem_bounds[0]) or torch.any(new_x_qehvi > problem_bounds[1]):
            print("Warning: Candidate solution out of problem bounds.", flush=True)

        # Endzeitpunkt der Iteration von mobo
        end_time = time.time()

        # Ausführungszeit der Iteration berechnen
        execution_time = end_time - start_time
        print(f"Iteration {iteration} execution time: {execution_time:.2f} seconds", flush=True)  # Logging iteration execution time
        data = [{'Optimization': iteration, 'Execution_Time': execution_time}]
        write_data_to_csv(csv_file_path, ['Optimization', 'Execution_Time'], data)

        # --- 
        # send and get new data from Unity:
        new_obj_qehvi = objective_function(new_x_qehvi[0])
        print(f"New objective values from Unity: {new_obj_qehvi}", flush=True)  # Logging new objective values

        # Check if new objective values are within the expected range
        if torch.any(new_obj_qehvi > 1) or torch.any(new_obj_qehvi < -1):
            print("Warning: New objective values are out of expected range [-1, 1].", flush=True)

        # Update training points
        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi.unsqueeze(0)])
        print(f"Updated training data. Train X shape: {train_x_qehvi.shape}, Train Obj shape: {train_obj_qehvi.shape}", flush=True)  # Logging training data update

        # Compute hypervolumes
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)
        #print(f"Computed hypervolume for iteration {iteration}: {volume}", flush=True)  # Logging hypervolume

        save_xy(train_x_qehvi, train_obj_qehvi, iteration)
        save_hypervolume_to_file(hvs_qehvi, iteration)

        # Reinitialize GP model with updated data
        #print("Reinitializing GP model with updated training data...", flush=True)  # Logging reinitialization
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    #-----------------------

    #-----------------------
    # ...also, plot the results
    #plot_pareto(train_x_qehvi, train_obj_qehvi, hvs_qehvi)
    #-----------------------

    #-----------------------
    # Tell Unity that the optimization has finished
    #-----------------------
    print("Send Data: optimization_finished,", flush=True)
    conn.sendall(bytes('optimization_finished,', 'utf-8'))
    #-----------------------

    return hvs_qehvi, train_x_qehvi, train_obj_qehvi



# -------------------------------------------------------
# -------------------------------------------------------
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# -------------------------------------------------------
# Save X and Y samples to CSV
# -------------------------------------------------------
def save_xy(x_sample, y_sample, iteration):
    # Define the CSV file path
    CURRENT_DIR = os.getcwd()  # Current working directory
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData", USER_ID)
    if not os.path.exists(PROJECT_PATH):
        os.makedirs(PROJECT_PATH)
    print("Project Path for Observations:", PROJECT_PATH, flush=True)

    # Detect pareto front points before denormalization (while the objective values are still all set to maximization)
    #pareto_mask = is_non_dominated(y_sample) # not used as it does not return identical rows as Pareto true
    pareto_mask = calculate_pareto_front(y_sample)

    # Convert tensors to numpy arrays
    x_csv = x_sample.clone().cpu().numpy()
    y_csv = y_sample.clone().cpu().numpy()

    # Denormalize parameter values for the last row
    for i in range(len(x_csv[-1])):
        x_csv[-1][i] = denormalize_to_original_param(x_csv[-1][i], parameters_info[i][0], parameters_info[i][1])

    # Denormalize objective values for the last row
    for i in range(len(y_csv[-1])):
        y_csv[-1][i] = denormalize_to_original_obj(y_csv[-1][i], objectives_info[i][0], objectives_info[i][1], objectives_info[i][2])

    # Combine the new record
    new_record = np.concatenate((y_csv[-1], x_csv[-1]))
    observations_csv_file_path = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    
    # Read current CSV data (if it exists)
    if os.path.exists(observations_csv_file_path):
        current_data = pd.read_csv(observations_csv_file_path, delimiter=';')
    else:
        current_data = pd.DataFrame(columns=['UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase', 'IsPareto'] + objective_names + parameter_names)
    
    # Create a DataFrame for the new record
    new_data = pd.DataFrame([[
        USER_ID,
        CONDITION_ID,
        GROUP_ID,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        iteration + N_INITIAL,
        'optimization',
        'FALSE',  # Placeholder for IsPareto, to be updated
        *new_record
    ]], columns=current_data.columns)

    # Append new data to the current dataset
    updated_data = pd.concat([current_data, new_data], ignore_index=True)

    # Update IsPareto column
    updated_data['IsPareto'] = ['TRUE' if is_pareto else 'FALSE' for is_pareto in pareto_mask]

    # Save updated data back to the CSV
    updated_data.to_csv(observations_csv_file_path, sep=';', index=False)
# -------------------------------------------------------
# -------------------------------------------------------

# -------------------------------------------------------
# Save hypervolume values to CSV
# -------------------------------------------------------
def save_hypervolume_to_file(hvs_qehvi, iteration):
    # Define the CSV file path
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData", USER_ID)
    if not os.path.exists(PROJECT_PATH):
        os.makedirs(PROJECT_PATH)
    print("Project Path for Hypervolumes:", PROJECT_PATH, flush=True)
    
    # Save hypervolume per evaluation
    hypervolume_value = np.array(hvs_qehvi)
    header_volume = ["Hypervolume", "Run"]
    hypervolume_csv_file_path = os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv")
    with open(hypervolume_csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if os.path.getsize(hypervolume_csv_file_path) == 0:
            writer.writerow(header_volume)
        writer.writerow([hypervolume_value[-1], iteration])
# -------------------------------------------------------
# -------------------------------------------------------


# -------------------------------------------------------
def calculate_pareto_front(obj_vals):
    """
    Calculate the Pareto front given objective values, returning a mask to indicate which points are Pareto-optimal.
    
    Args:
    obj_vals (torch.Tensor or np.ndarray): A 2D array where each row represents an observation and each column represents an objective.
    
    Returns:
    pareto_mask (torch.Tensor): A boolean mask indicating which rows are Pareto-optimal.
    """
    # Convert to torch tensor if input is not a tensor
    if not isinstance(obj_vals, torch.Tensor):
        obj_vals_tmp = torch.tensor(obj_vals, dtype=torch.float64)
    else:
        obj_vals_tmp = obj_vals.clone()

    #print("Input Objective Values:", obj_vals_tmp, flush=True)

    # Identify the number of objectives and observations
    num_observations = obj_vals_tmp.shape[0]
    pareto_mask = torch.ones(num_observations, dtype=torch.bool)

    # Check each observation if it is dominated by any other observation
    for i in range(num_observations):
        if pareto_mask[i]:
            current_values = obj_vals_tmp[i]
            # Set Pareto mask to False for dominated observations
            is_dominated = torch.all(obj_vals_tmp <= current_values, dim=1) & torch.any(obj_vals_tmp < current_values, dim=1)
            pareto_mask[is_dominated] = False

    return pareto_mask
# -------------------------------------------------------
# -------------------------------------------------------

# -------------------------------------------------------
# The following code can be used to generate plots for the Pareto Front and Hypervolume (however, only supports THREE objectives)
# -------------------------------------------------------
def plot_pareto(x_sample, y_sample, hvs_qehvi):
    CURRENT_DIR = os.getcwd()  # Current working directory
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData")
    
    # Detect pareto front points
    pareto_mask = calculate_pareto_front(y_sample)

    # Convert tensors to numpy arrays and denormalize objective values
    x_plot = x_sample.clone().cpu().numpy()
    y_plot = y_sample.clone().cpu().numpy()

    #print(f"y_sample shape: {y_sample.shape}, objectives_info length: {len(objectives_info)}", flush=True)

    # Denormalize objective values
    for i in range(y_plot.shape[1]):
        # Debugging: print the index and length of objectives_info to identify if it's out of range
        #print(f"Index i: {i}, Objectives Info Length: {len(objectives_info)}", flush=True)
        y_plot[:, i] = denormalize_to_original_obj(y_plot[:, i], objectives_info[i][0], objectives_info[i][1], objectives_info[i][2])

    # Denormalize parameter values
    for i in range(x_plot.shape[1]):
        # Debugging: print the index and length of parameters_info to identify if it's out of range
        #print(f"Index i: {i}, Parameters Info Length: {len(parameters_info)}", flush=True)
        x_plot[:, i] = denormalize_to_original_param(x_plot[:, i], parameters_info[i][0], parameters_info[i][1])
    
    pareto_obj = y_plot[pareto_mask]
    pareto_front = x_plot[pareto_mask]

    x_all = y_plot[:, 0]
    y_all = y_plot[:, 1]
    z_all = y_plot[:, 2]
    pareto_obj = pareto_obj[pareto_obj[:, 0].argsort()]
    x_pareto = pareto_obj[:, 0]
    y_pareto = pareto_obj[:, 1]
    z_pareto = pareto_obj[:, 2]

    # Create parallel coordinates plot
    line_index = list(range(len(pareto_front)))
    pareto_design_parameters = np.concatenate((np.array([line_index]).T, pareto_front), axis=1)

    columns_i = ["iter"]
    for i in range(len(pareto_design_parameters[0]) - 1):
        columns_i.append("x" + str(i + 1))
    
    design_parameters_pd = pd.DataFrame(data=pareto_design_parameters, index=line_index, columns=columns_i)
    
    plt.rcParams['figure.max_open_warning'] = 50
    plt.figure(figsize=(15, 6))

    # 3D Scatter plot for objective values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Objective values (3D Pareto Front)', fontsize=14)
    ax.scatter(x_all, y_all, z_all, label='All Points')
    ax.scatter(x_pareto, y_pareto, z_pareto, color='r', label='Pareto Front')
    ax.set_xlabel(objective_names[0])
    ax.set_ylabel(objective_names[1])
    ax.set_zlabel(objective_names[2])
    ax.set_xlim(objectives_info[0][0], objectives_info[0][1])
    ax.set_ylim(objectives_info[1][0], objectives_info[1][1])
    ax.set_zlim(objectives_info[2][0], objectives_info[2][1])
    ax.legend()
    plt.savefig(os.path.join(PROJECT_PATH, 'opt-process-pareto-img.svg'), format='svg', dpi=50)
    plt.clf()

    # Parallel coordinates plot for design parameters
    plt.figure(figsize=(15, 6))
    plt.title('Design parameters')
    pd.plotting.parallel_coordinates(design_parameters_pd, "iter")
    plt.savefig(os.path.join(PROJECT_PATH, 'opt-process-design-parameter-img.svg'), format='svg', dpi=50)
    plt.clf()

    # Hypervolume increase plot
    plt.figure()
    plt.plot(hvs_qehvi)
    plt.title("Pareto Hypervolume Increase", fontsize=24)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.savefig(os.path.join(PROJECT_PATH, 'opt-process-hyper-img.svg'), format='svg', dpi=50)
    plt.clf()
# -------------------------------------------------------
# -------------------------------------------------------
    
    
    
# Run the sampling and optimization loop, 
#   after receiving the initialization data (see "data = conn.recv(1024)" blocker at the beginning of this file):
hvs_qehvi, train_x_qehvi, train_obj_qehvi = mobo_execute(SEED, N_ITERATIONS, N_INITIAL)
