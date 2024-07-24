from import_all import *
import socket
import pickle
import pandas as pd
import time
import platform
import csv


HOST = ''
PORT = 56001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print('Server starts, waiting for connection...', flush=True)
conn, addr = s.accept()
print('Connected by', addr)
data = conn.recv(1024)


### iter, init_sample, design_parameter_num, objective_num
received = data.decode("utf-8").split('_')
parameter_raw = received[0].split('/')
print('Parameter', parameter_raw)
parameters_strinfo = []
parameters_info = []
for i in range(len(parameter_raw) ):
    parameters_strinfo.append(parameter_raw[i].split(','))
for strlist in parameters_strinfo:
    parameters_info.append(list(map(float, strlist)))

objective_raw = received[1].split('/')
print('Objective', objective_raw)
objectives_strinfo = []
objectives_info = []
for i in range(len(objective_raw)):
    objectives_strinfo.append(objective_raw[i].split(','))
for strlist in objectives_strinfo:
    objectives_info.append(list(map(float, strlist)))

print("Objectives info", len(objectives_info))

problem_dim = 16 #dimension of the parameters x
num_objs = 2 #dimension of the objectives y

#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference point in objective function space
#ref_point = torch.tensor([-1. for _ in range(num_objs)]).cuda()
ref_point = torch.tensor([-1. for _ in range(num_objs)]).to(device)
#print("Ref_point", ref_point)

# Design parameter bounds
problem_bounds = torch.zeros(2, problem_dim, **tkwargs)
#print("problem_bounds", problem_bounds)


# initialize the problem bounds
# for i in range(4):
#     problem_bounds[0][i] = parameters_info[i][0]
#     problem_bounds[1][i] = parameters_info[i][1]
problem_bounds[1] = 1

# print(problem_bounds)

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())


# Sample objective function
def objective_function(x_tensor):
    x = x_tensor.cpu().numpy()
    print("x", x)
    print("Parameters_info:", parameters_info)
    send_data = "parameters,"
    for i in range(len(x)):
        send_data += str(
            round((x[i]) * (parameters_info[i][1] - parameters_info[i][0]) + parameters_info[i][0], 3)) + ","

    send_data = send_data[:-2]
    print("Send Data: ", send_data )
    conn.sendall(bytes(send_data, 'utf-8'))

    data = conn.recv(1024)
    received_objective = []
    if data:
        received_objective = list(map(float, data.decode("utf-8").split(',')))
        print("data", received_objective)
    if len(data) == 0:
        print("unity end")
    if (len(received_objective) != num_objs):
        print("recevied objective number not consist")

    print("received: ", received_objective)


    def limit_range(f):
        if (f > 1):
            f = 1
        elif (f < -1):
            f = -1
        return f

    print("Received Objective", len(received_objective))



    fs = []
    # Normalization
    for i in range(num_objs):
        f = (received_objective[i] - objectives_info[i][0]) / (objectives_info[i][1] - objectives_info[i][0])
        f = f * 2 - 1
        if (objectives_info[i][2] == 1):
            f *= -1
        f = limit_range(f)
        fs.append(f)

    return torch.tensor(fs, dtype=torch.float64).to(device)
    #return torch.tensor(fs, dtype=torch.float64).cuda()

#das hier heißt dass die Optimierungsfunktion immer random beginnt und deshalb direkt mit der Applikation verbunden sein muss
# n_samples muss 2(d+1) wobei d = num_objs ist sein (https://botorch.org/tutorials/multi_objective_bo)
def generate_initial_data(n_samples=12):
    # generate training data
    train_x = draw_sobol_samples(
        bounds=problem_bounds, n=1, q=n_samples, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0)
    # train_obj = objective_function(train_x)

    train_obj = []
    for i, x in enumerate(train_x):
        print(f"initial sample: {i + 1}")
        train_obj.append(objective_function(x))

    train_obj_array = np.array([item.cpu().detach().numpy() for item in train_obj], dtype=np.float64)


    print("Shape der Arrays: ", train_x.shape, torch.tensor(train_obj_array).to(device).shape)
    #return train_x, torch.tensor([item.cpu().detach().numpy() for item in train_obj], dtype=torch.float64).cuda()
    return train_x, torch.tensor(train_obj_array).to(device)


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


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


def load_data():
    data = pd.read_csv('user_observations.csv')
    y = torch.tensor(np.array([data["Trust"].values, data["Understanding"].values, data["MentalLoad"].values,
                               data["PerceivedSafety"].values, data["Aesthetics"].values,
                               data["Acceptance"].values]).T).to(device)

    x = torch.tensor(
        np.array(
            [data["Trajectory"].values, data["TrajectoryAlpha"].values, data["TrajectorySize"].values, data["EgoTrajectory"].values, data["EgoTrajectoryAlpha"].values, data["EgoTrajectorySize"].values, data["PedestrianIntention"].values, data["PedestrianIntentionSize"].values, data["SemanticSegmentation"].values, data["SemanticSegmentationAlpha"].values, data["CarStatus"].values, data["CarStatusAlpha"].values, data["CoveredArea"].values, data["CoveredAreaAlpha"].values, data["CoveredSize"].values, data["OccludedCars"].value]).T).to(device)
    return x, y

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
        print("Fehler beim Erstellen der Datei:", str(e))



def write_data_to_csv(csv_file_path, fieldnames, data):
    try:
        with open(csv_file_path, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writerows(data)
    except Exception as e:
        print("Fehler beim Schreiben der Datei:", str(e))


def mobo_execute(seed, iterations, initial_samples):
    torch.manual_seed(seed)

    hv = Hypervolume(ref_point=ref_point)
    # Hypervolumes
    hvs_qehvi = []

    # Initial Samples
    # train_x_qehvi, train_obj_qehvi = load_data()
    train_x_qehvi, train_obj_qehvi = generate_initial_data(n_samples=initial_samples)

    #train_x_qehvi = train_x_qehvi.cpu()
    #train_obj_qehvi = train_obj_qehvi.cpu()

    # Initialize GP models
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

    # Compute Pareto front and hypervolume
    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]
    volume = hv.compute(pareto_y)
    hvs_qehvi.append(volume)
    save_xy(train_x_qehvi, train_obj_qehvi, hvs_qehvi, 0)

    print("Y:")

    fieldnames = ['Exploitation', 'Execution_Time']

    current_dir = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    #parent_dir = os.path.dirname(os.path.dirname(current_dir))
    parent_dir = os.path.dirname(current_dir)

    filename = 'ExecutionTimes.csv'
    if platform.system() == "Windows":
        if "UNITY_EDITOR" in os.environ:
            # logdata_path = os.path.join(current_dir, "LogData")
            logdata_path = os.path.join(current_dir, "LogData")
        else:
            logdata_path = os.path.join(current_dir, "Project_Data", "Data", "LogData")
    elif platform.system() == "Darwin":  # macOS
        logdata_path = os.path.join(parent_dir, "Data/LogData")

    # Pfad zur CSV-Datei festlegen
    csv_file_path = os.path.join(logdata_path, filename)
    # CSV-Datei erstellen
    create_csv_file(csv_file_path, fieldnames)

    # Go through the iterations
    for iteration in range(1, iterations + 1):
        print("Iteration: " + str(iteration))
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
        data = [{'Exploitation': iteration, 'Execution_Time': execution_time}]
        write_data_to_csv(csv_file_path, fieldnames, data)

        new_obj_qehvi = objective_function(new_x_qehvi[0])

        # Update training points
        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi.unsqueeze(0)])

        # Compute hypervolumes
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)

        save_xy(train_x_qehvi, train_obj_qehvi, hvs_qehvi, iteration)
        # print("mask", pareto_mask)
        # print("pareto y", pareto_y)
        # print("volume", volume)

        # print("trianing x", train_x_qehvi)
        # print("trianing obj", train_obj_qehvi)

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

    return hvs_qehvi, train_x_qehvi, train_obj_qehvi




def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_iterations_to_csv1(csv_file_path, iteration):
    # Lese die vorhandene CSV-Datei ein
    df = pd.read_csv(csv_file_path, delimiter=';')

    # Überprüfe, ob der Header "Run" bereits vorhanden ist
    if 'Run' not in df.columns:
        # Füge den Header "Run" hinzu und fülle mit NaN-Werten
        df['Run'] = pd.Series([pd.NA] * len(df))

    # Iteriere über die Zeilen des DataFrames
    for i, row in df.iterrows():
        if pd.isna(row['Run']):
            # Füge die Iteration in leere Zellen ein
            df.at[i, 'Run'] = iteration[i] if i < len(iteration) else pd.NA

    # Speichere den aktualisierten DataFrame in die CSV-Datei
    df.to_csv(csv_file_path, index=False, sep=';')



def save_iterations_to_csv(csv_file_path, iteration):
    # Überprüfen, ob die Iteration bereits in der CSV-Datei vorhanden ist
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if len(row) > 0 and row[0] == str(iteration):
                return  # Iteration bereits vorhanden, nichts tun

    # Wenn die Iteration nicht gefunden wurde, füge sie zur CSV-Datei hinzu
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([iteration])  # Nur die Iteration hinzufügen, weitere Spalten können angepasst werden


def save_xy(x_sample, y_sample, hvs_qehvi, iteration):
    #directory = '../Data/LogData'
    #if not os.path.exists(directory):
    #    os.makedirs(directory)

    current_dir = os.getcwd()  # Aktuelles Arbeitsverzeichnis
    #parent_dir = os.path.dirname(os.path.dirname(current_dir))
    parent_dir = os.path.dirname(current_dir)
    if platform.system() == "Windows":
        if "UNITY_EDITOR" in os.environ:
            # project_path = os.path.join(current_dir, "LogData")
            project_path = os.path.join(current_dir, "LogData")
        else:
            # project_path = os.path.join(current_dir, "Project_Data\Data\LogData")
            project_path = os.path.join(current_dir, "Project_Data", "Data", "LogData")
    elif platform.system() == "Darwin":  # macOS
        project_path = os.path.join(parent_dir, "Data/LogData")

    print("Project Path:", project_path)

    # Detect pareto front points
    pareto_mask = is_non_dominated(y_sample)
    pareto_obj = y_sample[pareto_mask]

    x_sample = x_sample.cpu().numpy()
    y_sample = y_sample.cpu().numpy()
    pareto_obj = pareto_obj.cpu().numpy()
    pareto_front = x_sample[pareto_mask.cpu()]

    all_record = np.concatenate((y_sample, x_sample), axis=1)

    f_values = y_sample.copy()
    f_values = np.array([list(x) for x in f_values])

    x_all = f_values[:, 0]
    y_all = f_values[:, 1]
    pareto_obj = pareto_obj[pareto_obj[:, 0].argsort()]
    x_pareto = pareto_obj[:, 0]
    y_pareto = pareto_obj[:, 1]

    # Create parallel coordinates plot

    line_index = list(range(len(pareto_front)))
    pareto_design_parameters = np.concatenate((np.array([line_index]).T, pareto_front), axis=1)

    columns_i = ["iter"]
    for i in range(len(pareto_design_parameters[0]) - 1):
        columns_i.append("x" + str(i + 1))

    design_parameters_pd = pd.DataFrame(data=pareto_design_parameters, index=line_index, columns=columns_i)

    #plt.rcParams['figure.max_open_warning'] = 50
    #plt.figure(figsize=(15, 6))
    # plt.figure()

    #plt.subplot(121)
    #plt.title('Objective values')
    #plt.scatter(x_all, y_all)
    #plt.plot(x_pareto, y_pareto, color='r')
    #plt.xlabel('Completion Time')
    #plt.ylabel('Accuracy')
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    # plt.savefig('../Assets/Resources/opt-process-parato-img', dpi=50)

    # plt.clf()

    # plt.figure()

    #plt.subplot(122)
    #plt.title('Design parameters')
    #pd.plotting.parallel_coordinates(design_parameters_pd, "iter")
    # Save plot
    #plt.savefig('../Assets/Resources/opt-process-design-parameter-img', dpi=50)
    #plt.savefig('imgs/observations', dpi=50)
    #plt.clf()
    # plt.show()
    #plt.figure()
    #plt.plot(hvs_qehvi)
    #plt.title("Pareto Hypervolume Increase", fontsize=24)
    #plt.tick_params(axis='x', labelsize=16)
    #plt.tick_params(axis='y', labelsize=16)
    #plt.savefig('../Assets/Resources/opt-process-hyper-img', dpi=50)
    #plt.savefig('imgs/hypervolume', dpi=50)
    #plt.clf()

    # add new column to identify pareto points
    index_arr = []
    for i in pareto_mask:
        temp = ""
        if (i):
            temp = "TRUE"
        else:
            temp = "FALSE"
        index_arr.append(temp)
    # np.savetxt('../Assets/Resources/VROptimizer.csv', all_record, delimiter=',', fmt="%s")

    #QEHVI misst die Qualität der Lösung (QEHVI steht für Quality Estimate Hypervolume Improvement). Es gibt an, wie gut die aktuelle Lösung im Vergleich zu den bisherigen Lösungen ist.
    #Zielfunktionswerte: Diese Werte repräsentieren die Leistung oder Qualität der Lösung basierend auf den Zielfunktionen.
    #Designparameterwerte: Diese Werte repräsentieren die Einstellungen oder Konfigurationen der Parameter, die optimiert werden sollen.
    #areto-Dominanz-Indikator: Dieser Wert gibt an, ob eine bestimmte Lösung Pareto-dominiert ist oder nicht. Pareto-Dominanz bedeutet, dass eine Lösung in mindestens einer Zielfunktion besser ist als eine andere Lösung, ohne in einer anderen Zielfunktion schlechter zu sein.
    all_record = np.concatenate((all_record, np.array([index_arr]).T), axis=1)

    header = np.array(['Trust QEHVI','Understanding QEHVI', 'MentalLoad QEHVI', 'PerceivedSafety QEHVI', 'Aesthetics QEHVI', 'Acceptance QEHVI', 'Trajectory', 'TrajectoryAlpha', 'TrajectorySize', 'EgoTrajectory', 'EgoTrajectoryAlpha', 'EgoTrajectorySize', 'PedestrianIntention ', 'PedestrianIntentionSize ', 'SemanticSegmentation', 'SemanticSegmentationAlpha', 'CarStatus', 'CarStatusAlpha', 'CoveredArea', 'CoveredAreaAlpha', 'CoveredAreaSize', 'OccludedCars', 'IsPareto'])

    header_run = np.append(header, "Run")  # Fügt den Header "Run" hinzu
    """
    observations_csv_file_path = os.path.join(project_path, "ObservationsPerEvaluation.csv")
    #np.savetxt(observations_csv_file_path, all_record_append, delimiter=';', fmt="%s")

    # Hinzufügen der Daten zur CSV-Datei
    with open(observations_csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if os.path.getsize(observations_csv_file_path) == 0:
            writer.writerow(header_run)
        if iteration == 0:
            writer.writerows([[*row, iteration] for row in all_record])
        else:
            writer.writerow([*all_record[-1], iteration])
    

    #if platform.system() == "Windows":
        #np.savetxt(project_path + '\ObservationsPerEvaluation.csv', all_record, delimiter=';', fmt="%s")
        #save_iterations_to_csv(project_path + '\ObservationsPerEvaluation.csv', iteration)
    #elif platform.system() == "Darwin":  # macOS
    #    np.savetxt(project_path + '/ObservationsPerEvaluation.csv', all_record, delimiter=';', fmt="%s")

    #save_iterations_to_csv(project_path + '/ObservationsPerEvaluation.csv', iteration)
    #Die Variable hvs_qehvi enthält eine Liste von Werten, die den Hypervolume-Wert des Pareto-Fronts zu jedem Optimierungsschritt darstellen
    #Der Hypervolume-Wert ist ein Maß für die Ausdehnung und Diversität des Pareto-Fronts im Zielraum.
    #Ein höherer Hypervolume-Wert deutet auf eine bessere Qualität der gefundenen Pareto-Front hin.
    hypervolume_value = np.array(hvs_qehvi)
    header_volume = ["Hypervolume", "Run"]

    # Save hypervolume values to CSV file
    hypervolume_csv_file_path = os.path.join(project_path, "HypervolumePerEvaluation.csv")

    # Hinzufügen der Hypervolume-Daten
    with open(hypervolume_csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if os.path.getsize(hypervolume_csv_file_path) == 0:
            writer.writerow(header_volume)
        writer.writerow([hypervolume_value[-1], iteration])
    """
    #if platform.system() == "Windows":
    #    np.savetxt(project_path + '\HypervolumePerEvaluation.csv', hypervolume_value, delimiter=';',
    #               header=header_volume, comments='')
    #    save_iterations_to_csv(project_path + '\HypervolumePerEvaluation.csv', iteration)
    #elif platform.system() == "Darwin":  # macOS
    #    np.savetxt(project_path + '/HypervolumePerEvaluation.csv', hypervolume_value, delimiter=';',
    #               header=header_volume, comments='')
    #    save_iterations_to_csv(project_path + '/HypervolumePerEvaluation.csv', iteration)

hvs_qehvi, train_x_qehvi, train_obj_qehvi = mobo_execute(SEED, N_ITERATIONS, N_INITIAL)

