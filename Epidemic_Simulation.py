#============================Pakage imports============================#
import pandas as pd
import numpy as np
from collections import deque
import csv
import math
import os
import time
import json
from datetime import datetime, timedelta  
from collections import defaultdict
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
Project_path = os.path.join(os.path.dirname(__file__))
pd.set_option('display.max_columns', None)

#========================== Define Seeding Events ===========================#
def generate_seeding_events():
    seeding_date = params['seeding_date']
    num_exposed = params['Number_of_Initial_Exposed']
    if num_exposed <= 0:
        return {}

    top_nodes = sorted(g_node_list, key=lambda n: n.initial_population, reverse=True)
    if not top_nodes:
        return {}

    np.random.seed(params['random_seed'])
    max_candidates = min(100, len(top_nodes))
    chosen_node = np.random.choice(top_nodes[:max_candidates])

    return {
        seeding_date: {
            chosen_node.node_id: {"exposed": num_exposed}
        }
    }


def generate_daily_imported_cases(date):
    daily_imported = params.get('Daily_Imported_Cases')
    imported_nodes_num = params.get('Number_of_Imported_Nodes')
    if daily_imported <= 0 or imported_nodes_num <= 0:
        return {}

    date_obj = datetime.strptime(date, "%Y-%m-%d")
    seeding_date_obj = datetime.strptime(params['seeding_date'], "%Y-%m-%d")

    # First lockdown window
    lockdown_start_obj = datetime.strptime("2020-03-31", "%Y-%m-%d")  # first day of "no imports" period

    # End of lockdown, when imports resume
    lockdown_end_obj = datetime.strptime("2020-08-01", "%Y-%m-%d")    # imports allowed from this day on


    if not (
        (seeding_date_obj < date_obj < lockdown_start_obj)
        or (date_obj >= lockdown_end_obj)
    ):
        return {}

    # Sort nodes by population
    top_nodes = sorted(g_node_list, key=lambda n: n.initial_population, reverse=True)

    # Population array (avoid zeros to keep probability non-zero)
    populations = np.array([max(n.initial_population, 1) for n in top_nodes], dtype=float)
    pop_weights = populations / populations.sum()

    # Seed per day for reproducibility
    np.random.seed(hash(date) % (2**32))

    # Choose imported_nodes_num nodes based on population weighting (no repeats for same day)
    imported_indices = np.random.choice(
        len(top_nodes),
        size=imported_nodes_num,
        replace=False,
        p=pop_weights
    )
    imported_nodes = [top_nodes[i] for i in imported_indices]

    # Calculate distribution weights for selected nodes
    sel_populations = np.array([max(n.initial_population, 1) for n in imported_nodes], dtype=float)
    sel_weights = sel_populations / sel_populations.sum()

    # Assign imported cases proportionally to population
    imported_counts = np.floor(sel_weights * daily_imported).astype(int)

    # Handle rounding remainder
    remainder = daily_imported - imported_counts.sum()
    if remainder > 0:
        for i in np.argsort(-sel_weights)[:remainder]:
            imported_counts[i] += 1

    # Build dictionary
    imported_dict = {
        node.node_id: {"exposed": int(count)}
        for node, count in zip(imported_nodes, imported_counts)
        if count > 0
    }

    return imported_dict


#========================== Define Safe Sampling Method =========================#

def safe_sample_compartments(available_dict, agent_num):
    """
    Assign agent_num individuals to SEIR compartments.
    Uses multinomial sampling, respects upper bounds from available_dict.
    """
    # Safely filter and convert input
    filtered = {}
    for k, v in available_dict.items():
        try:
            v_float = float(v)
            if v_float > 0 and np.isfinite(v_float):
                filtered[k] = int(v_float)
        except (ValueError, TypeError):
            continue  # skip invalid entries

    total_available = sum(filtered.values())
    if total_available == 0:
        return {}

    states = list(filtered.keys())
    weights = np.array([filtered[k] for k in states], dtype=np.float64)
    probs = weights / weights.sum()

    # Sample all assignments at once
    assigned_states = np.random.choice(states, size=agent_num, p=probs)
    assignment_counts = dict(Counter(assigned_states))

    # Trim to not exceed actual availability
    for state in assignment_counts:
        assignment_counts[state] = min(assignment_counts[state], filtered[state])

    return assignment_counts


#========================== Define the function to process each node in the first pass =========================#
def process_node_first_pass(args):
    node_id, agents_data, date, raw_compartments = args

    skipped_ids = set()
    agent_results = []

    # Step 0: Validate compartment availability
    total_pop = sum(raw_compartments.values())
    if total_pop == 0:
        skipped_ids.update(agent_id for agent_id, _ in agents_data)
        return node_id, agent_results, skipped_ids, {}

    # Step 1: Filter and prepare compartments
    valid_compartments = {k: int(np.floor(v)) for k, v in raw_compartments.items() if v >= 1}
    total_available = sum(valid_compartments.values())
    if total_available == 0:
        skipped_ids.update(agent_id for agent_id, _ in agents_data)
        return node_id, agent_results, skipped_ids, {}

    # Step 2: Total agents needed for this node
    total_needed = sum(agent_num for _, agent_num in agents_data)
    if total_needed == 0:
        return node_id, agent_results, skipped_ids, {}

    # Step 3: Batched sampling based on compartment proportions
    all_assignments = safe_sample_compartments(valid_compartments, min(total_needed, total_available))
    if not all_assignments or sum(all_assignments.values()) == 0:
        skipped_ids.update(agent_id for agent_id, _ in agents_data)
        return node_id, agent_results, skipped_ids, {}

    # Flatten assignments and shuffle
    flat_list = [state for state, count in all_assignments.items() for _ in range(count)]
    np.random.shuffle(flat_list)

    # Step 4: Slice and assign to each agent
    offset = 0
    state_deductions = {k: 0 for k in valid_compartments}
    for agent_id, agent_num in agents_data:
        if offset >= len(flat_list):
            skipped_ids.add(agent_id)
            continue

        assign_size = min(agent_num, len(flat_list) - offset)
        agent_slice = flat_list[offset : offset + assign_size]
        assignment = dict(Counter(agent_slice))
        total_assigned = sum(assignment.values())

        if total_assigned == 0:
            skipped_ids.add(agent_id)
        else:
            agent_results.append((agent_id, assignment, total_assigned))
            for state, count in assignment.items():
                state_deductions[state] += count

        offset += assign_size

    return node_id, agent_results, skipped_ids, state_deductions



#====================== Define needed dictionaries and lists =======================#
g_internal_node_seq_dict = {} # Dictionary to map node_id to internal node sequence
g_internal_node_pair_dict = {} # Dictionary to map (from_node_id, to_node_id) to internal link sequence
g_node_list = [] # List to store all nodes
g_agent_list = [] # List to store all agents
g_date_agent_dict  = {} # Dictionary to store agents by departure date
SEEDING_EVENTS = {}   # placeholder
agent_exposure_pressure = {}


#========================== Define Node Class =========================#
class Node:
    def __init__(self, node_id, node_seq, x_coord, y_coord,population):
        self.node_id = node_id
        self.node_seq = node_seq
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.inflow = {}  # Inflow of agents to the node
        self.outflow = {}  # Outflow of agents from the node

        # Initial SEIR Compartments
        initial_population = max(0, int(float(population))) # ensure non-negative integer
        self.initial_population = initial_population 
        if initial_population > 0:
            self.susceptible = {"2020-01-01": initial_population}
            self.exposed = {"2020-01-01": 0}
        else:
            self.susceptible = {"2020-01-01": 0}
            self.exposed = {"2020-01-01": 0}
        self.asymptomatic_infected = {"2020-01-01": 0}
        self.presymptomatic_infected = {"2020-01-01": 0}
        self.infected = {"2020-01-01": 0}
        self.recovered = {"2020-01-01": 0}
        self.total_agents = {"2020-01-01": initial_population}


#========================== Define Agent Class =========================#
class Agent:
    def __init__(self, agent_id, link_seq, from_node_id, to_node_id,
                 departure_date, agent_num,if_start=False, if_end=False):
        self.agent_id = agent_id  
        self.trip_seq = int(link_seq)  # trip index in the chain
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.departure_date = departure_date
        self.if_start = if_start
        self.if_end = if_end
        self.agent_num = int(agent_num)


#========================== Define Network Class and SEIR Simulation =========================#
# Network class to handle SEIR simulation and agent movement
class Network:
    def __init__(self):
        pass
    def seir_simulation(self, iteration, date):
        SEIR_start = time.time()  # Start timer for this date
        agent_assignment_cache = {}

        date_obj = datetime.strptime(date, "%Y-%m-%d")
        prev_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
        pop_scale_exp = 0.5
        national_avg_contact_rate = params['average_contact_rate']
        national_avg_contact_rate_home = params['average_contact_rate_home']
        national_avg_population = params['national_avg_population']

        # === 0) Add imported cases if after seeding date ===
        seeding_date_obj = datetime.strptime(params['seeding_date'], "%Y-%m-%d")
        if date_obj > seeding_date_obj:
            daily_imports = generate_daily_imported_cases(date)
        else:
            daily_imports = {}

        # === 1) Apply node-level SEIR transitions ===
        for node in g_node_list:

            if prev_date in SEEDING_EVENTS and node.node_id in SEEDING_EVENTS[prev_date]:
                seed = SEEDING_EVENTS[prev_date][node.node_id]

                S_y = node.susceptible[prev_date]
                E_y = node.exposed[prev_date]
                I_y = node.infected[prev_date]

                exp_seed = seed.get("exposed", 0)
                inf_seed = seed.get("infected", 0)
                required = exp_seed + inf_seed

                node.exposed[prev_date] = E_y + exp_seed
                node.infected[prev_date] = I_y + inf_seed
                node.susceptible[prev_date] = S_y - required

                node.susceptible[date] = node.susceptible[prev_date]
                node.exposed[date] = node.exposed[prev_date]
                node.infected[date] = node.infected[prev_date]
                node.asymptomatic_infected[date] = node.asymptomatic_infected.get(prev_date, 0)
                node.presymptomatic_infected[date] = node.presymptomatic_infected.get(prev_date, 0)
                node.recovered[date] = node.recovered.get(prev_date, 0)
                node.total_agents[date] = node.total_agents.get(prev_date, 0)
            else:
                node.susceptible[date] = node.susceptible.get(prev_date, node.susceptible.get("2020-01-01"))
                node.exposed[date] = node.exposed.get(prev_date, node.exposed.get("2020-01-01"))
                node.asymptomatic_infected[date] = node.asymptomatic_infected.get(prev_date, 0)
                node.presymptomatic_infected[date] = node.presymptomatic_infected.get(prev_date, 0)
                node.infected[date] = node.infected.get(prev_date, 0)
                node.recovered[date] = node.recovered.get(prev_date, 0)
                node.total_agents[date] = node.total_agents.get(prev_date, 0)

            # === Inject daily imports (AFTER copying values from yesterday) ===
            if node.node_id in daily_imports:
                imp_seed = daily_imports[node.node_id]
                node.exposed[date] += imp_seed.get("exposed", 0)
                node.infected[date] += imp_seed.get("infected", 0)
                # node.susceptible[date] -= imp_seed.get("infected", 0)
                # if node.susceptible[date] < 0:
                #     node.susceptible[date] = 0

            # Negative check
            if any(x < 0 for x in [
                node.susceptible[date], node.exposed[date],
                node.asymptomatic_infected[date], node.presymptomatic_infected[date],
                node.infected[date], node.recovered[date]
            ]):
                raise ValueError(f"[ERROR] Negative SEIR values at node {node.node_id} on {date}")

            # Skip SEIR transition if no population
            prev_total = node.total_agents.get(prev_date, 0)
            if prev_total == 0:
                continue

            # Infectious pressure
            denominator = prev_total
            if denominator > 0:
                population_scaling = (denominator / national_avg_population) ** pop_scale_exp
                #scaled_contact_rate = national_avg_contact_rate_home * population_scaling
                infectious_pressure = (
                    national_avg_contact_rate_home * (
                        params["BETA_1"] * node.infected[date] +
                        params["BETA_2"] * node.asymptomatic_infected[date] +
                        params["BETA_3"] * node.presymptomatic_infected[date]
                    )
                ) / denominator
            else:
                infectious_pressure = 0

            lookup_key = (str(node.node_id), str(prev_date))
            agent_pressure = agent_exposure_pressure.get(lookup_key, 0)
            infectious_pressure = max(agent_pressure, infectious_pressure)

            if not np.isfinite(infectious_pressure) or infectious_pressure < 0:
                raise ValueError(f"[ERROR] infectious_pressure={infectious_pressure:.5f} at node {node.node_id} on {date} is invalid.")

            # Transitions
            new_exposed = node.susceptible[date] * infectious_pressure
            new_asymptomatic = params["RHO"] * node.exposed[date] / params["LAMBDA"]
            new_presymptomatic = (1 - params["RHO"]) * node.exposed[date] / params["LAMBDA"]
            new_infected = node.presymptomatic_infected[date] / params["ALPHA"]
            new_recovered_A = node.asymptomatic_infected[date] / params["GAMMA_A"]
            new_recovered_I = node.infected[date] / params["GAMMA_I"]

            if any(x < 0 for x in [
                new_exposed, new_asymptomatic, new_presymptomatic,
                new_infected, new_recovered_A, new_recovered_I
            ]):
                raise ValueError(f"[ERROR] Negative SEIR transition values at node {node.node_id} on {date}")

            node.susceptible[date] -= new_exposed
            node.exposed[date] += new_exposed - new_asymptomatic - new_presymptomatic
            node.asymptomatic_infected[date] += new_asymptomatic - new_recovered_A
            node.presymptomatic_infected[date] += new_presymptomatic - new_infected
            node.infected[date] += new_infected - new_recovered_I
            node.recovered[date] += new_recovered_A + new_recovered_I

            if any(x < 0 for x in [
                node.susceptible[date], node.exposed[date],
                node.asymptomatic_infected[date], node.presymptomatic_infected[date],
                node.infected[date], node.recovered[date]
            ]):
                raise ValueError(f"[ERROR] Negative SEIR values at node {node.node_id} on {date}")

        SEIR_Node_1 = time.time()
        #print(f"[DEBUG]NODE INIT on {date} took {SEIR_Node_1 - SEIR_start:.2f} seconds")
        
        
        
        # === First Pass: Assign SEIR state to agents starting today ===

        skipped_agent_ids = set()
        agents_today = g_date_agent_dict.get(date, [])
        agent_lookup = {a.agent_id: a for a in agents_today if a.if_start}

        # Group agents by origin node (only IDs + nums passed to subprocess)
        node_to_agents = defaultdict(list)
        for agent in agent_lookup.values():
            node_to_agents[agent.from_node_id].append((agent.agent_id, agent.agent_num))

        # Prepare node-level tasks
        node_tasks = []
        for node_id, agent_data_list in node_to_agents.items():
            from_node = g_node_list[g_internal_node_seq_dict[node_id]]
            raw_compartments = {
                'susceptible': from_node.susceptible.get(date, 0),
                'exposed': from_node.exposed.get(date, 0),
                'asymptomatic_infected': from_node.asymptomatic_infected.get(date, 0),
                'presymptomatic_infected': from_node.presymptomatic_infected.get(date, 0),
                'infected': from_node.infected.get(date, 0),
                'recovered': from_node.recovered.get(date, 0),
            }

            if any(x < 0 for x in raw_compartments.values()):
                raise ValueError(f"[ERROR] Negative SEIR values at node {node_id} on {date}")

            node_tasks.append((node_id, agent_data_list, date, raw_compartments))

        # Run node-level agent assignment in parallel
        agent_assignment_cache = {}
        state_deductions_by_node = {}

        with ProcessPoolExecutor() as executor:
            results = executor.map(process_node_first_pass, node_tasks)
            for node_id, agent_results, skipped_ids, state_deductions in results:
                skipped_agent_ids.update(skipped_ids)
                state_deductions_by_node[node_id] = state_deductions

                for agent_id, assignment_counts, total_assigned in agent_results:
                    agent_assignment_cache[agent_id] = assignment_counts
                    agent_lookup[agent_id].agent_num = total_assigned

        # Apply SEIR deductions to from_node
        for node_id, deductions in state_deductions_by_node.items():
            from_node = g_node_list[g_internal_node_seq_dict[node_id]]
            for state, count in deductions.items():
                from_node.__dict__[state][date] -= count
                if from_node.__dict__[state][date] < 0:
                    raise ValueError(f"[ERROR] Negative {state} at node {node_id} after applying batch deduction on {date}")

        # Filter out skipped agents
        g_date_agent_dict[date] = [a for a in agents_today if a.agent_id not in skipped_agent_ids]
        SEIR_Node_2 = time.time()  # End timer for this date
        #print(f"[DEBUG]First Pass on {date} took {SEIR_Node_2 - SEIR_Node_1:.2f} seconds")



        # === Second Pass: Process link-level movement and exposure ===
        node_arrivals_today = {}
        agents_today = sorted(g_date_agent_dict.get(date), key=lambda a: a.trip_seq)  # Sort for sequence

        for agent in agents_today:

            from_node = g_node_list[g_internal_node_seq_dict[agent.from_node_id]]
            to_node = g_node_list[g_internal_node_seq_dict[agent.to_node_id]]

            from_node.outflow[date] = from_node.outflow.get(date, 0) + agent.agent_num
            to_node.inflow[date] = to_node.inflow.get(date, 0) + agent.agent_num

            if to_node.node_id not in node_arrivals_today:
                node_arrivals_today[to_node.node_id] = {'I': 0, 'A': 0, 'P': 0, 'N': 0}

            node_arrivals_today[to_node.node_id]['N'] += agent.agent_num

            # Use actual group composition from assignment
            assignment_counts = agent_assignment_cache.get(agent.agent_id, None)
            if assignment_counts:
                for state, count in assignment_counts.items():
                    if state == 'infected':
                        node_arrivals_today[to_node.node_id]['I'] += count
                    elif state == 'asymptomatic_infected':
                        node_arrivals_today[to_node.node_id]['A'] += count
                    elif state == 'presymptomatic_infected':
                        node_arrivals_today[to_node.node_id]['P'] += count
            else:
                raise ValueError(f"[ERROR] Agent {agent.agent_id} has no assignment counts on {date}.")
                

        # === Exposure step ===
        for agent in agents_today:
            assignment_counts = agent_assignment_cache.get(agent.agent_id)
            start_status = dict(assignment_counts)
            susceptible_count = assignment_counts.get('susceptible')
            if susceptible_count is None:
                susceptible_count = 0
            exposed_count = 0

            if susceptible_count > 0:
                to_node = g_node_list[g_internal_node_seq_dict[agent.to_node_id]]

                S_existing = to_node.susceptible.get(date)
                E_existing = to_node.exposed.get(date)
                I_existing = to_node.infected.get(date)
                A_existing = to_node.asymptomatic_infected.get(date)
                P_existing = to_node.presymptomatic_infected.get(date)
                R_existing = to_node.recovered.get(date)

                arrivals = node_arrivals_today.get(agent.to_node_id)
                I_arrive = arrivals['I']
                A_arrive = arrivals['A']
                P_arrive = arrivals['P']
                N_arrive = arrivals['N']
                I = I_existing + I_arrive
                A = A_existing + A_arrive
                P = P_existing + P_arrive
                N = S_existing + E_existing + I_existing + A_existing + P_existing + R_existing+ N_arrive

                # Avoid division by zero and invalid values
                if N > 0:
                    population_scaling = (N / national_avg_population) ** pop_scale_exp
                    scaled_contact_rate = national_avg_contact_rate * population_scaling
                    exposed_rate = (
                        national_avg_contact_rate * (
                            params['BETA_1'] * I +
                            params['BETA_2'] * A +
                            params['BETA_3'] * P
                        )
                    ) / N
                else:
                    exposed_rate = 0.0
                if not np.isfinite(exposed_rate) or exposed_rate < 0 or exposed_rate > 1:
                    raise ValueError(f"[ERROR] Invalid exposed_rate={exposed_rate} for agent {agent.agent_id} on {date}")
                agent_exposure_pressure[(str(agent.to_node_id), date)] = exposed_rate
                # Sample how many become exposed from susceptible subgroup
                exposed_count = np.random.binomial(susceptible_count, exposed_rate)

                # Update assignment
                assignment_counts['susceptible'] -= exposed_count
                assignment_counts['exposed'] = assignment_counts.get('exposed',0) + exposed_count

                # Save updated composition
                agent_assignment_cache[agent.agent_id] = assignment_counts

            else:
                exposed_rate = 0.0  # no susceptibles, no exposure

            # Compose transfer_dict using updated group status
            total_group = sum(assignment_counts.values())
            transfer_dict = {
                'S': assignment_counts.get('susceptible',0) / total_group,
                'E': assignment_counts.get('exposed',0) / total_group,
                'A': assignment_counts.get('asymptomatic_infected',0) / total_group,
                'P': assignment_counts.get('presymptomatic_infected',0) / total_group,
                'I': assignment_counts.get('infected',0) / total_group,
                'R': assignment_counts.get('recovered',0) / total_group
            }

        # === Third Pass: At end of trip chain, deposit agents ===
        for agent in g_date_agent_dict.get(date):
            if agent.if_end:
                assignment_counts = agent_assignment_cache.get(agent.agent_id)
                if assignment_counts:
                    to_node = g_node_list[g_internal_node_seq_dict[agent.to_node_id]]
                    for state, count in assignment_counts.items():
                        if state in ['susceptible', 'exposed', 'asymptomatic_infected', 'presymptomatic_infected', 'infected', 'recovered']:
                            to_node.__dict__[state][date] = to_node.__dict__[state].get(date, 0) + count
                    #print(f"[DEPOSITED] Agent {agent.agent_id} returned {assignment_counts} to {to_node.node_id} on {date}")
                else:
                    raise ValueError(f"[ERROR] Agent {agent.agent_id} has no assignment counts on {date}.")
        for node in g_node_list:
            node.total_agents[date] = (
                node.susceptible[date] + node.exposed[date] +
                node.asymptomatic_infected[date] + node.presymptomatic_infected[date] +
                node.infected[date] + node.recovered[date]
            )

def data_reading():
    with open("nodes_Lagos_population_full.csv") as file:
        csv_reader = csv.DictReader(file)
        node_seq = 0
        for row in csv_reader:
            population = row.get('population_pixel_assigned', '0')
            node = Node(
                row['node_id'], node_seq, row.get('x_coord', '0'), row.get('y_coord', '0'),population
            )
            g_node_list.append(node)
            g_internal_node_seq_dict[row['node_id']] = node_seq
            node_seq += 1
    print(f"Loaded {len(g_node_list)} nodes from nodes file")
    # Function to load agents from a file
    def load_agents_parquet(parquet_file):
        usecols = ['Agent_id', 'link_seq', 'from_node_id', 'to_node_id', 'Date_dep', 'if_start', 'if_end', 'agent_num']
        df = pd.read_parquet(parquet_file, columns=usecols)

        # Filter rows before loop
        df = df[df['from_node_id'].isin(g_internal_node_seq_dict) & df['to_node_id'].isin(g_internal_node_seq_dict)]

        agents = []
        date_dict = defaultdict(list)

        for row in df.itertuples(index=False):
            agent = Agent(
                agent_id=row.Agent_id,
                link_seq=row.link_seq,
                from_node_id=row.from_node_id,
                to_node_id=row.to_node_id,
                departure_date=row.Date_dep,
                if_start=bool(row.if_start),
                if_end=bool(row.if_end),
                agent_num=row.agent_num if hasattr(row, 'agent_num') else 1,
            )
            date_dict[agent.departure_date].append(agent)
            agents.append(agent)

        # Batch update globals
        for dep_date, agent_list in date_dict.items():
            g_date_agent_dict.setdefault(dep_date, []).extend(agent_list)
        g_agent_list.extend(agents)


    # Load agent files
    load_agents_parquet("LBS_agent.parquet")



def Simulation():
    network = Network()
    

    min_date = min(g_date_agent_dict.keys())
    max_date = max(g_date_agent_dict.keys())
    all_dates = [ (datetime.strptime(min_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
              for i in range((datetime.strptime(max_date, "%Y-%m-%d") - datetime.strptime(min_date, "%Y-%m-%d")).days + 1) ]
    # Run simulation for multiple days, tracking each agent's progress
    for date in all_dates:
        start_time_date = time.time()
        network.seir_simulation(0, date)
        #print(f"SEIR simulation for {date} completed.")
        end_time_date = time.time()
        elapsed_date = end_time_date - start_time_date
        #print(f"SEIR simulation for {date} completed in {elapsed_date:.2f} seconds ({elapsed_date/60:.2f} minutes)")
    # Write node dynamics including SEIR compartments + total agents
    with open('node_dynamics_geohash.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Add "total_agents" to the header
        writer.writerow([
            'node_id', 'date','inflow', 'outflow',
            'susceptible', 'exposed', 'asymptomatic_infected', 'presymptomatic_infected',
            'infected', 'recovered', 'total_agents'
        ])

        for node in g_node_list:
            for date in all_dates:
                writer.writerow([
                    node.node_id, date,
                    node.inflow.get(date, 0),  # Fix
                    node.outflow.get(date, 0),  # Fix
                    node.susceptible.get(date, 0), node.exposed.get(date, 0),
                    node.asymptomatic_infected.get(date, 0), node.presymptomatic_infected.get(date, 0),
                    node.infected.get(date, 0), node.recovered.get(date, 0),
                    node.total_agents.get(date, 0)
                ])

    #pd.DataFrame(trajectory_rows).to_csv("agent_trajectories.csv", index=False)

    print("Node dynamics saved to node_dynamics.csv")




if __name__ == '__main__':
    import json
    # === Load params.json if it exists ===
    if os.path.exists("params.json"):
        with open("params.json", "r") as f:
            params = json.load(f)

    # Set seed after loading params
    np.random.seed(params['random_seed'])
    start_time = time.time()
    print("Starting SEIR simulation...")
    print(f"Using parameters: {params}")
    print("Loading data...")
    data_reading()
    end_time_reading = time.time()
    data_reading_elapsed= end_time_reading - start_time
    print(f"Data reading completed in {data_reading_elapsed:.2f} seconds ({data_reading_elapsed/60:.2f} minutes)")
    SEEDING_EVENTS = generate_seeding_events()
    Simulation()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n✅ Simulation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
