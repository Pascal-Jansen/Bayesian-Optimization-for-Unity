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

N_INITIAL = 5
N_ITERATIONS = 10  # Number of optimization iterations

BATCH_SIZE = 1  # Number of design parameter points to query at next iteration
NUM_RESTARTS = 10  # Used for the acquisition function number of restarts in optimization
RAW_SAMPLES = 1024  # Durch höhere RawSamples kein OptimierungsFehler (Optimization failed within `scipy.optimize.minimize` with status 1.')
MC_SAMPLES = 512  # Number of samples to approximate acquisition function
SEED = 3  # Seed to initialize the initial samples obtained

PROBLEM_DIM = 16 #dimension of the parameters x
NUM_OBJS = 2 #dimension of the objectives y

WARM_START = False #true if there is initial data (accsible from the following paths) that should be used before optimization restarts
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
print('Parameter', parameter_raw)
parameters_strinfo = []
parameters_info = []
for i in range(len(parameter_raw) ):
    parameters_strinfo.append(parameter_raw[i].split(','))
for strlist in parameters_strinfo:
    parameters_info.append(list(map(float, strlist)))

objective_raw = init_data[3].split('/')
print('Objective', objective_raw)
objectives_strinfo = []
objectives_info = []
for i in range(len(objective_raw)):
    objectives_strinfo.append(objective_raw[i].split(','))
for strlist in objectives_strinfo:
    objectives_info.append(list(map(float, strlist)))

print("Objectives info", len(objectives_info))

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
#ref_point = torch.tensor([-1. for _ in range(num_objs)]).cuda()
ref_point = torch.tensor([-1. for _ in range(NUM_OBJS)]).to(device)
#print("Ref_point", ref_point)

# Design parameter bounds
problem_bounds = torch.zeros(2, PROBLEM_DIM, **tkwargs)
#print("problem_bounds", problem_bounds)

# initialize the problem bounds
# for i in range(4):
#     problem_bounds[0][i] = parameters_info[i][0]
#     problem_bounds[1][i] = parameters_info[i][1]
problem_bounds[1] = 1

# print(problem_bounds)

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

# -------------------------------------------------------
# Sample current objective function from the Unity application
# -------------------------------------------------------
#  do this in EACH iteration of the defined total optimization iterations (= n_samples + n_iterations)
def objective_function(x_tensor):
    x = x_tensor.cpu().numpy()
    print("x", x)
    print("Parameters_info:", parameters_info)
    send_data = "parameters,"
    # Denormalize x_numpy and obj_numpy
    x_denormalized = np.array([denormalize_to_original(x[i], parameters_info[i][0], parameters_info[i][1]) for i in range(len(x))])
    # prepare parameter string parameters:
    for i in range(len(x_denormalized)):
        send_data += str(
            round((x_denormalized[i]) * (parameters_info[i][1] - parameters_info[i][0]) + parameters_info[i][0], 3)) + ","
    # send parameter data to Unity:
    send_data = send_data[:-2]
    print("Send Data: ", send_data )
    conn.sendall(bytes(send_data, 'utf-8'))

    # wait for Unity to respond ...

    data = conn.recv(1024)
    received_objective = []
    if data:
        received_objective = list(map(float, data.decode("utf-8").split(',')))
        #print("data", received_objective)
    if len(data) == 0:
        print("unity end")
    if (len(received_objective) != NUM_OBJS):
        print("recevied objective number not consistent")

    print("Received Objective Values: ", received_objective)

    def limit_range(f):
        if (f > 1):
            f = 1
        elif (f < -1):
            f = -1
        return f

    print("Received Number of Objectives", len(received_objective))

    fs = []
    # Normalization
    for i in range(NUM_OBJS):
        f = (received_objective[i] - objectives_info[i][0]) / (objectives_info[i][1] - objectives_info[i][0])
        f = f * 2 - 1
        if (objectives_info[i][2] == 1):
            f *= -1
        f = limit_range(f)
        fs.append(f)

    return torch.tensor(fs, dtype=torch.float64).to(device)
    #return torch.tensor(fs, dtype=torch.float64).cuda()


def denormalize_to_original(value, lower_bound, upper_bound):
    return lower_bound + (value + 1) / 2 * (upper_bound - lower_bound)

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
        print("Error creating file:", str(e))

# -------------------------------------------------------
# write_data_to_csv
# -------------------------------------------------------
def write_data_to_csv(csv_file_path, fieldnames, data):
    try:
        with open(csv_file_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writerows(data)
    except Exception as e:
        print("Error writing to file:", str(e))

# -------------------------------------------------------
# generate_initial_data
# -------------------------------------------------------
#das hier heißt dass die Optimierungsfunktion immer random beginnt und deshalb direkt mit der Applikation verbunden sein muss
# n_samples muss 2(d+1) wobei d = num_objs ist sein (https://botorch.org/tutorials/multi_objective_bo)
def generate_initial_data(n_samples=14):
    
    # Define the CSV file path
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData")
    OBSERVATIONS_LOG_PATH = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")

    # Check if the file exists and write the header if it does not
    if not os.path.exists(OBSERVATIONS_LOG_PATH):
        header = np.array(['User_ID', 'Condition_ID', 'Group_ID', 'Timestamp', 'Run', 'Phase', 'IsPareto'] + objective_names + parameter_names)
        with open(OBSERVATIONS_LOG_PATH, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(header)

    # Generate training data
    train_x = draw_sobol_samples(
        bounds=problem_bounds, n=1, q=n_samples, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0)

    # loop to sample objective values from the users to these training data
    train_obj = []
    for i, x in enumerate(train_x):
        print(f"----------------------Initial Sample: {i + 1}")
        obj = objective_function(x)
        train_obj.append(obj)

        # Convert objective and parameter values to numpy arrays for logging
        x_numpy = x.cpu().numpy()
        obj_numpy = obj.cpu().detach().numpy()
        # Denormalize x_numpy and obj_numpy
        x_denormalized = np.array([denormalize_to_original(x_numpy[i], parameters_info[i][0], parameters_info[i][1]) for i in range(len(x_numpy))])
        obj_denormalized = np.array([denormalize_to_original(obj_numpy[i], objectives_info[i][0], objectives_info[i][1]) for i in range(len(obj_numpy))])
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
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# -------------------------------------------------------
# optimize_qehvi
# -------------------------------------------------------
def optimize_qehvi(model, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=problem_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem_bounds)
    return new_x

# -------------------------------------------------------
# mobo_execute ... this is the main method that handles the sampling and optimization loops
# -------------------------------------------------------
def mobo_execute(seed, iterations, initial_samples):
    #-----------------------
    # prepare file logging 
    #-----------------------
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData")
    os.makedirs(PROJECT_PATH, exist_ok=True)
    csv_file_path = os.path.join(PROJECT_PATH, 'ExecutionTimes.csv')
    create_csv_file(csv_file_path, ['Optimization', 'Execution_Time'])
    #-----------------------

    #-----------------------
    # init mobo
    #-----------------------
    torch.manual_seed(seed)

    hv = Hypervolume(ref_point=ref_point)
    # Hypervolumes
    hvs_qehvi = []

    # Whether the optimizer explore first with initial data (warm-start) or random points in the parameter space...
    if WARM_START:
        train_x_qehvi, train_obj_qehvi = load_data()
    else:
        train_x_qehvi, train_obj_qehvi = generate_initial_data(n_samples=initial_samples)

    # Initialize GP models
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

    # Compute and save hypervolume for the sampling phase before optimization
    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]
    volume = hv.compute(pareto_y)
    hvs_qehvi.append(volume)
    save_hypervolume_to_file(hvs_qehvi, 0)
    #-----------------------

    #-----------------------
    # Go through the iterations
    #-----------------------
    for iteration in range(1, iterations + 1):
        print("----------------------Iteration: " + str(iteration))
        # Startzeitpunkt der Iteration von mobo
        start_time = time.time()
        # Fit Models
        fit_gpytorch_mll(mll_qehvi)
        # Define qEI acquisition modules using QMC sampler
        sample_shape = torch.Size([MC_SAMPLES])
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=sample_shape, seed = SEED)
        # Endzeitpunkt der Iteration aufzeichnen
        # Optimize acquisition functions and get new observations
        new_x_qehvi = optimize_qehvi(model_qehvi, train_obj_qehvi, qehvi_sampler)
        # Endzeitpunkt der Iteration von mobo
        end_time = time.time()

        # Ausführungszeit der Iteration berechnen
        execution_time = end_time - start_time
        data = [{'Optimization': iteration, 'Execution_Time': execution_time}]
        write_data_to_csv(csv_file_path, ['Optimization', 'Execution_Time'], data)

        new_obj_qehvi = objective_function(new_x_qehvi[0]) # data will be sent to Unity in this function

        # Update training points
        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi.unsqueeze(0)])

        # Compute hypervolumes
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)

        save_xy(train_x_qehvi, train_obj_qehvi, iteration)
        save_hypervolume_to_file(hvs_qehvi, iteration)
        # print("mask", pareto_mask)
        # print("pareto y", pareto_y)
        # print("volume", volume)

        # print("trianing x", train_x_qehvi)
        # print("trianing obj", train_obj_qehvi)

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    #-----------------------

    #-----------------------
    # Tell Unity that the optimization has finished
    #-----------------------
    print("Send Data: ", 'optimization_finished,' )
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
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData")
    print("Project Path for Observations:", PROJECT_PATH)

    # Detect pareto front points
    pareto_mask = is_non_dominated(y_sample)

    # Convert tensors to numpy arrays
    x_sample = x_sample.cpu().numpy()
    y_sample = y_sample.cpu().numpy()

    # Denormalize parameter values for the last row
    for i in range(len(x_sample[-1])):
        x_sample[-1][i] = denormalize_to_original(x_sample[-1][i], parameters_info[i][0], parameters_info[i][1])

    # Denormalize objective values for the last row
    for i in range(len(y_sample[-1])):
        y_sample[-1][i] = denormalize_to_original(y_sample[-1][i], objectives_info[i][0], objectives_info[i][1])

    # Combine all records and pareto identification
    all_record = np.concatenate((y_sample, x_sample), axis=1)
    index_arr = ["TRUE" if i else "FALSE" for i in pareto_mask]
    all_record = np.concatenate((np.array([index_arr]).T, all_record), axis=1)

    # Save observations per evaluation
    observations_csv_file_path = os.path.join(PROJECT_PATH, "ObservationsPerEvaluation.csv")
    header = np.array(['User_ID', 'Condition_ID', 'Group_ID', 'Timestamp', 'Run', 'Phase', 'IsPareto'] + objective_names + parameter_names)
    with open(observations_csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if os.path.getsize(observations_csv_file_path) == 0:
            writer.writerow(header)
        writer.writerow([USER_ID, CONDITION_ID, GROUP_ID, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iteration, 'optimization', *all_record[-1]])  # Only write the last record
# -------------------------------------------------------
# -------------------------------------------------------

# -------------------------------------------------------
# Save hypervolume values to CSV
# -------------------------------------------------------
def save_hypervolume_to_file(hvs_qehvi, iteration):
    # Define the CSV file path
    CURRENT_DIR = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    PROJECT_PATH = os.path.join(CURRENT_DIR, "LogData")
    print("Project Path for Hypervolumes:", PROJECT_PATH)
    
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

# Run the sampling and optimization loop, 
#   after receiving the initialization data (see "data = conn.recv(1024)" blocker at the beginning of this file):
hvs_qehvi, train_x_qehvi, train_obj_qehvi = mobo_execute(SEED, N_ITERATIONS, N_INITIAL)


### The following code can be used to generate plots for the Pareto Front and Hypervolume (however, only supports two objectives)
    # Detect pareto front points
    #pareto_mask = is_non_dominated(y_sample)
    #pareto_obj = y_sample[pareto_mask]

    #x_sample = x_sample.cpu().numpy()
    #y_sample = y_sample.cpu().numpy()
    #pareto_obj = pareto_obj.cpu().numpy()
    #pareto_front = x_sample[pareto_mask.cpu()]

    #all_record = np.concatenate((y_sample, x_sample), axis=1)

    #f_values = y_sample.copy()
    #f_values = np.array([list(x) for x in f_values])

    #x_all = f_values[:, 0]
    #y_all = f_values[:, 1]
    #pareto_obj = pareto_obj[pareto_obj[:, 0].argsort()]
    #x_pareto = pareto_obj[:, 0]
    #y_pareto = pareto_obj[:, 1]

    # Create parallel coordinates plot
    #line_index = list(range(len(pareto_front)))
    #pareto_design_parameters = np.concatenate((np.array([line_index]).T, pareto_front), axis=1)

    #columns_i = ["iter"]
    #for i in range(len(pareto_design_parameters[0]) - 1):
    #    columns_i.append("x" + str(i + 1))
    
    #design_parameters_pd = pd.DataFrame(data=pareto_design_parameters, index=line_index, columns=columns_i)
    
    #plt.rcParams['figure.max_open_warning'] = 50
    #plt.figure(figsize=(15, 6))

    #plt.subplot(121)
    #plt.title('Objective values')
    #plt.scatter(x_all, y_all)
    #plt.plot(x_pareto, y_pareto, color='r')
    #plt.xlabel('Completion Time')
    #plt.ylabel('Accuracy')
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    #plt.savefig(os.path.join(project_path, 'opt-process-parato-img.png'), dpi=50)

    #plt.clf()

    #plt.subplot(122)
    #plt.title('Design parameters')
    #pd.plotting.parallel_coordinates(design_parameters_pd, "iter")
    #plt.savefig(os.path.join(project_path, 'opt-process-design-parameter-img.png'), dpi=50)
    #plt.clf()

    #plt.figure()
    #plt.plot(hvs_qehvi)
    #plt.title("Pareto Hypervolume Increase", fontsize=24)
    #plt.tick_params(axis='x', labelsize=16)
    #plt.tick_params(axis='y', labelsize=16)
    #plt.savefig(os.path.join(project_path, 'opt-process-hyper-img.png'), dpi=50)
    #plt.clf()