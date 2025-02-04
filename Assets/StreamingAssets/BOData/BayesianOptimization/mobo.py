from matplotlib import pyplot as plt
from import_all import *
import socket
import pickle
import pandas as pd
import time
import platform
import csv
import os  # For working with file paths and directories

# Define default values for global variables
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

CURRENT_DIR = ""
# PROJECT_PATH will be set uniquely in mobo_execute
PROJECT_PATH = ""
OBSERVATIONS_LOG_PATH = ""

N_INITIAL = 5  # Number of sampling iterations
N_ITERATIONS = 10  # Number of optimization iterations

BATCH_SIZE = 1  # Number of design parameter points to query at the next iteration
NUM_RESTARTS = 10  # Used for the acquisition function number of restarts in optimization
RAW_SAMPLES = 1024  # Prevents optimization failures in `scipy.optimize.minimize`
MC_SAMPLES = 512  # Number of samples to approximate acquisition function
SEED = 3  # Seed to initialize the initial samples obtained

PROBLEM_DIM = 16  # Dimension of the parameters (x)
NUM_OBJS = 2      # Dimension of the objectives (y)

WARM_START = False  # True if there is initial data (accessible from the following paths) that should be used before optimization restarts
CSV_PATH_PARAMETERS = ""
CSV_PATH_OBJECTIVES = ""

USER_ID = ""
CONDITION_ID = ""
GROUP_ID = ""

### Helper function to create a unique folder
def get_unique_folder(parent, folder_name):
    """
    Creates a folder under 'parent' with name 'folder_name'.
    If it already exists, appends "_1", "_2", etc., until a unique name is found.
    Returns the full path to the created folder.
    """
    base_path = os.path.join(parent, folder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path
    counter = 1
    while True:
        new_path = os.path.join(parent, f"{folder_name}_{counter}")
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        counter += 1

### Init socket and start receiving data
HOST = ''
PORT = 56001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print('Server starts, waiting for connection...', flush=True)
conn, addr = s.accept()
print('Connected by', addr, flush=True)

# Wait for data ...
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

### Parse parameter and objective info
parameter_raw = init_data[2].split('/')
print('Parameter', parameter_raw, flush=True)
parameters_strinfo = []
parameters_info = []
for i in range(len(parameter_raw)):
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
parameter_names = [param_name for param_name in parameter_names_raw]

objective_names_raw = init_data[5].split(',')
objective_names = [obj_name for obj_name in objective_names_raw]

### Init user information
study_info_raw = init_data[6].split(',')
USER_ID = study_info_raw[0]
CONDITION_ID = study_info_raw[1]
GROUP_ID = study_info_raw[2]

### Init the device and other settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference point in objective function space
# This tells the optimizer that all objectives should be maximized, having a -1 to 1 range (higher is better)
ref_point = torch.tensor([-1. for _ in range(NUM_OBJS)]).to(device)

# Design parameter bounds to range from 0 to 1
problem_bounds = torch.zeros(2, PROBLEM_DIM, **tkwargs)
problem_bounds[1] = 1  # Set upper bounds

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

# -------------------------------------------------------
# Sample the current objective function from the Unity application
# -------------------------------------------------------
# This is done in EACH iteration of the defined total optimization iterations (= n_samples + n_iterations)
def objective_function(x_tensor):
    x = x_tensor.cpu().numpy()
    # Denormalize x_numpy using the provided function
    x_denormalized = np.array([denormalize_to_original_param(x[i], parameters_info[i][0], parameters_info[i][1]) for i in range(len(x))])
    send_data = "parameters,"
    # Prepare parameter string for sending to Unity
    for i in range(len(x_denormalized)):
        send_data += str(round(x_denormalized[i], 3)) + ","
    send_data = send_data[:-1]  # Remove trailing comma
    print("Send Data: ", send_data, flush=True)
    conn.sendall(bytes(send_data, 'utf-8'))

    # Wait for Unity to respond
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

    # Normalize objective values to the range [-1, 1]
    fs = []
    for i in range(NUM_OBJS):
        f = (received_objective[i] - objectives_info[i][0]) / (objectives_info[i][1] - objectives_info[i][0])
        f = f * 2 - 1
        if objectives_info[i][2] == 1:  # Invert if the objective is set to "smaller is better"
            f *= -1
        fs.append(limit_range(f))

    print("Objective normalized:", fs)
    return torch.tensor(fs, dtype=torch.float64).to(device)

# Parameter denormalization (from normalized [0,1] to original scale)
def denormalize_to_original_param(value, lower_bound, upper_bound):
    result = lower_bound + value * (upper_bound - lower_bound)
    return np.round(result, 2) if isinstance(result, (float, int)) else np.round(result, 2)

# Objective denormalization (from normalized [-1,1] to original scale)
def denormalize_to_original_obj(value, lower_bound, upper_bound, smaller_is_better):
    if smaller_is_better == 1:  # Invert if objective was set to minimize
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
    global PROJECT_PATH
    CURRENT_DIR = os.getcwd()  # Current working directory
    # Use the global PROJECT_PATH (set in mobo_execute) instead of re-computing it
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    if not os.path.exists(OBSERVATIONS_LOG_PATH):
        header = np.array(['UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase', 'IsPareto'] + objective_names + parameter_names)
        with open(OBSERVATIONS_LOG_PATH, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(header)

    # Generate training data using Sobol samples
    train_x = draw_sobol_samples(
        bounds=problem_bounds, n=1, q=n_samples, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0)
    print("Initial training data (Sobol samples) in normalized range [-1, 1]:", train_x, flush=True)

    train_obj = []
    for i, x in enumerate(train_x):
        print(f"----------------------Initial Sample: {i + 1}", flush=True)
        obj = objective_function(x)
        train_obj.append(obj)

        x_numpy = x.cpu().numpy()
        obj_numpy = obj.cpu().detach().numpy()
        x_denormalized = np.array([denormalize_to_original_param(x_numpy[i], parameters_info[i][0], parameters_info[i][1]) for i in range(len(x_numpy))])
        obj_denormalized = np.array([denormalize_to_original_obj(obj_numpy[i], objectives_info[i][0], objectives_info[i][1], objectives_info[i][2]) for i in range(len(obj_numpy))])
        all_record = np.concatenate(([USER_ID], [CONDITION_ID], [GROUP_ID],
                                     [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())],
                                     [i+1], ['sampling'], ['FALSE'], obj_denormalized, x_denormalized))
        with open(OBSERVATIONS_LOG_PATH, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(all_record)

    train_obj_array = np.array([item.cpu().detach().numpy() for item in train_obj], dtype=np.float64)
    return train_x, torch.tensor(train_obj_array).to(device)

# -------------------------------------------------------
# load_data (if there are prior observations)
# -------------------------------------------------------
# (each row in the CSV represents one sampling iteration)
def load_data():
    CURRENT_DIR = os.getcwd()
    objective_path = os.path.join(CURRENT_DIR, "InitData", CSV_PATH_OBJECTIVES)
    parameter_path = os.path.join(CURRENT_DIR, "InitData", CSV_PATH_PARAMETERS)
    data_objectives = pd.read_csv(objective_path, delimiter=';')
    data_parameter = pd.read_csv(parameter_path, delimiter=';')
    y = torch.tensor(data_objectives.values, dtype=torch.float64).to(device)
    x = torch.tensor(data_parameter.values, dtype=torch.float64).to(device)
    return x, y

# -------------------------------------------------------
# initialize_model
# -------------------------------------------------------
def initialize_model(train_x, train_obj):
    x_min = train_x.min(dim=0)[0]
    x_max = train_x.max(dim=0)[0]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    train_x_normalized = (train_x - x_min) / x_range
    # Standardize objectives manually using the entire dataset
    train_obj_mean = train_obj.mean(dim=0)
    train_obj_std = train_obj.std(dim=0)
    train_obj_std[train_obj_std == 0] = 1.0
    train_obj_standardized = (train_obj - train_obj_mean) / train_obj_std
    model = SingleTaskGP(train_x_normalized, train_obj_standardized)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# -------------------------------------------------------
# optimize_qehvi
# -------------------------------------------------------
def optimize_qehvi(model, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=sampler,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=problem_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # Optionally add small random noise to encourage exploration:
    # candidates += torch.randn_like(candidates) * 0.01
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    return new_x

# -------------------------------------------------------
# mobo_execute ... Main optimization loop
# -------------------------------------------------------
def mobo_execute(seed, iterations, initial_samples):
    global PROJECT_PATH, OBSERVATIONS_LOG_PATH
    #-----------------------
    # Prepare file logging: create a unique folder for this run
    #-----------------------
    CURRENT_DIR = os.getcwd()
    base_folder = os.path.join(CURRENT_DIR, "LogData")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    # Create a unique folder for this run based on USER_ID
    PROJECT_PATH = get_unique_folder(base_folder, USER_ID)
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    csv_file_path = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(csv_file_path, ['Optimization', 'Execution_Time'])
    #-----------------------

    #-----------------------
    # Initialize MOBO
    #-----------------------
    torch.manual_seed(seed)
    hv = Hypervolume(ref_point=ref_point)
    hvs_qehvi = []

    if WARM_START:
        print("Loading warm start data...", flush=True)
        train_x_qehvi, train_obj_qehvi = load_data()
    else:
        print("Generating initial training data...", flush=True)
        train_x_qehvi, train_obj_qehvi = generate_initial_data(n_samples=initial_samples)

    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    print(f"Initial train_x_qehvi shape: {train_x_qehvi.shape}, train_obj_qehvi shape: {train_obj_qehvi.shape}", flush=True)

    if torch.any(train_obj_qehvi > 1) or torch.any(train_obj_qehvi < -1):
        print("Warning: Initial objective values are out of expected range [-1, 1].", flush=True)

    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]
    volume = hv.compute(pareto_y)
    hvs_qehvi.append(volume)
    save_hypervolume_to_file(hvs_qehvi, 0)
    #-----------------------

    for iteration in range(1, iterations + 1):
        print(f"----------------------MOBO Iteration: {iteration}", flush=True)
        start_time = time.time()
        fit_gpytorch_mll(mll_qehvi)

        if torch.any(torch.isnan(list(model_qehvi.parameters())[0].clone().detach())):
            print("Warning: NaN detected in GP model parameters after fitting.", flush=True)

        sample_shape = torch.Size([MC_SAMPLES])
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=sample_shape, seed=SEED)
        new_x_qehvi = optimize_qehvi(model_qehvi, train_obj_qehvi, qehvi_sampler)
        if torch.any(new_x_qehvi < problem_bounds[0]) or torch.any(new_x_qehvi > problem_bounds[1]):
            print("Warning: Candidate solution out of problem bounds.", flush=True)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Iteration {iteration} execution time: {execution_time:.2f} seconds", flush=True)
        data = [{'Optimization': iteration, 'Execution_Time': execution_time}]
        write_data_to_csv(csv_file_path, ['Optimization', 'Execution_Time'], data)

        new_obj_qehvi = objective_function(new_x_qehvi[0])
        print(f"New objective values from Unity: {new_obj_qehvi}", flush=True)
        if torch.any(new_obj_qehvi > 1) or torch.any(new_obj_qehvi < -1):
            print("Warning: New objective values are out of expected range [-1, 1].", flush=True)

        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi.unsqueeze(0)])
        print(f"Updated training data. Train X shape: {train_x_qehvi.shape}, Train Obj shape: {train_obj_qehvi.shape}", flush=True)

        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)
        save_xy(train_x_qehvi, train_obj_qehvi, iteration)
        save_hypervolume_to_file(hvs_qehvi, iteration)

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    #-----------------------

    # Optionally plot the results:
    # plot_pareto(train_x_qehvi, train_obj_qehvi, hvs_qehvi)

    print("Send Data: optimization_finished,", flush=True)
    conn.sendall(bytes('optimization_finished,', 'utf-8'))
    return hvs_qehvi, train_x_qehvi, train_obj_qehvi

# -------------------------------------------------------
# Utility functions for saving/loading objects remain unchanged
# -------------------------------------------------------
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# -------------------------------------------------------
# Save X and Y samples to CSV (uses global PROJECT_PATH)
# -------------------------------------------------------
def save_xy(x_sample, y_sample, iteration):
    global PROJECT_PATH
    # Use the global PROJECT_PATH set in mobo_execute
    observations_csv_file_path = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    print("Project Path for Observations:", PROJECT_PATH, flush=True)

    pareto_mask = calculate_pareto_front(y_sample)
    x_csv = x_sample.clone().cpu().numpy()
    y_csv = y_sample.clone().cpu().numpy()

    for i in range(len(x_csv[-1])):
        x_csv[-1][i] = denormalize_to_original_param(x_csv[-1][i], parameters_info[i][0], parameters_info[i][1])
    for i in range(len(y_csv[-1])):
        y_csv[-1][i] = denormalize_to_original_obj(y_csv[-1][i], objectives_info[i][0], objectives_info[i][1], objectives_info[i][2])

    new_record = np.concatenate((y_csv[-1], x_csv[-1]))
    if os.path.exists(observations_csv_file_path):
        current_data = pd.read_csv(observations_csv_file_path, delimiter=';')
    else:
        current_data = pd.DataFrame(columns=['UserID', 'ConditionID', 'GroupID', 'Timestamp', 'Iteration', 'Phase', 'IsPareto'] + objective_names + parameter_names)

    new_data = pd.DataFrame([[
        USER_ID,
        CONDITION_ID,
        GROUP_ID,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        iteration + N_INITIAL,
        'optimization',
        'FALSE',  # Placeholder for IsPareto; will be updated below
        *new_record
    ]], columns=current_data.columns)

    updated_data = pd.concat([current_data, new_data], ignore_index=True)
    updated_data['IsPareto'] = ['TRUE' if is_pareto else 'FALSE' for is_pareto in pareto_mask]
    updated_data.to_csv(observations_csv_file_path, sep=';', index=False)

# -------------------------------------------------------
# Save hypervolume values to CSV (uses global PROJECT_PATH)
# -------------------------------------------------------
def save_hypervolume_to_file(hvs_qehvi, iteration):
    global PROJECT_PATH
    hypervolume_csv_file_path = os.path.join(PROJECT_PATH, "HypervolumePerEvaluation.csv")
    print("Project Path for Hypervolumes:", PROJECT_PATH, flush=True)
    hypervolume_value = np.array(hvs_qehvi)
    header_volume = ["Hypervolume", "Run"]
    with open(hypervolume_csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if os.path.getsize(hypervolume_csv_file_path) == 0:
            writer.writerow(header_volume)
        writer.writerow([hypervolume_value[-1], iteration])

# -------------------------------------------------------
# Calculate Pareto front given objective values
# -------------------------------------------------------
def calculate_pareto_front(obj_vals):
    """
    Calculate the Pareto front given objective values, returning a mask to indicate which points are Pareto-optimal.
    
    Args:
        obj_vals (torch.Tensor or np.ndarray): A 2D array where each row represents an observation and each column an objective.
    
    Returns:
        pareto_mask (torch.Tensor): A boolean mask indicating which rows are Pareto-optimal.
    """
    if not isinstance(obj_vals, torch.Tensor):
        obj_vals_tmp = torch.tensor(obj_vals, dtype=torch.float64)
    else:
        obj_vals_tmp = obj_vals.clone()

    num_observations = obj_vals_tmp.shape[0]
    pareto_mask = torch.ones(num_observations, dtype=torch.bool)

    for i in range(num_observations):
        if pareto_mask[i]:
            current_values = obj_vals_tmp[i]
            is_dominated = torch.all(obj_vals_tmp <= current_values, dim=1) & torch.any(obj_vals_tmp < current_values, dim=1)
            pareto_mask[is_dominated] = False

    return pareto_mask

# -------------------------------------------------------
# Plot Pareto front and hypervolume (optional)
# -------------------------------------------------------
def plot_pareto(x_sample, y_sample, hvs_qehvi):
    global PROJECT_PATH
    CURRENT_DIR = os.getcwd()
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData")
    
    pareto_mask = calculate_pareto_front(y_sample)
    x_plot = x_sample.clone().cpu().numpy()
    y_plot = y_sample.clone().cpu().numpy()

    for i in range(y_plot.shape[1]):
        y_plot[:, i] = denormalize_to_original_obj(y_plot[:, i], objectives_info[i][0], objectives_info[i][1], objectives_info[i][2])
    for i in range(x_plot.shape[1]):
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

    line_index = list(range(len(pareto_front)))
    pareto_design_parameters = np.concatenate((np.array([line_index]).T, pareto_front), axis=1)
    columns_i = ["iter"] + [f"x{i+1}" for i in range(len(pareto_design_parameters[0]) - 1)]
    design_parameters_pd = pd.DataFrame(data=pareto_design_parameters, index=line_index, columns=columns_i)
    
    plt.rcParams['figure.max_open_warning'] = 50
    plt.figure(figsize=(15, 6))
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

    plt.figure(figsize=(15, 6))
    plt.title('Design parameters')
    pd.plotting.parallel_coordinates(design_parameters_pd, "iter")
    plt.savefig(os.path.join(PROJECT_PATH, 'opt-process-design-parameter-img.svg'), format='svg', dpi=50)
    plt.clf()

    plt.figure()
    plt.plot(hvs_qehvi)
    plt.title("Pareto Hypervolume Increase", fontsize=24)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.savefig(os.path.join(PROJECT_PATH, 'opt-process-hyper-img.svg'), format='svg', dpi=50)
    plt.clf()

# -------------------------------------------------------
# Run the sampling and optimization loop
# -------------------------------------------------------
hvs_qehvi, train_x_qehvi, train_obj_qehvi = mobo_execute(SEED, N_ITERATIONS, N_INITIAL)