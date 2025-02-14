import random
import numpy as np
from deap import base, creator, tools, algorithms

# Environment Parameters
NUMBER_OF_NURSES = 5
NUMBER_OF_DAYS = 7
SHIFTS_PER_DAY = 3

# Constraints
REQUIRED_NURSES_PER_SHIFT = 2
MAX_CONSECUTIVE_SHIFTS = 4
POPULATION_SIZE = 100

# Fitness Properties
UNDER_STAFF_PENALTY = 5
OVER_STAFF_PENALTY = 5
CONSECUTIVE_SHIFT_PENALTY = 8
PREFERENCE_PENALTY = 3

# penalty weights
under_staffing_w1 = 1
over_staffing_w2 = 1
consecutive_shift_w3 = 2
preference_violation_w4 = 1

# GA properties
GENERATIONS = 200

# SA properties
SA_INITIAL_TEMP = 5000
SA_COOLING_RATE = 0.995

# ASO Properties
NUMBER_OF_ANTS = 10
ITERATIONS = 100
EVAPORATION_RATE = 0.5
ALPHA = 1
BETA = 2

# PSO Properties
NUMBER_OF_PARTICLES = 20

#Tabu Search
TABU_SIZE = 10

nurse_preferences = np.random.randint(0, SHIFTS_PER_DAY, (NUMBER_OF_NURSES, NUMBER_OF_DAYS))
def evaluate_schedule(schedule):
    penalty = 0
    schedule = np.reshape(schedule, (NUMBER_OF_NURSES, NUMBER_OF_DAYS))

    nurses_on_shift = np.zeros((NUMBER_OF_DAYS, SHIFTS_PER_DAY))
    for day in range(NUMBER_OF_DAYS):
        for shift in range(SHIFTS_PER_DAY):
            nurses_on_shift[day][shift] = sum(1 for nurse in range(NUMBER_OF_NURSES) if schedule[nurse][day] == shift)

    understaff_penalty = 0
    overstaff_penalty = 0
    for day in range(NUMBER_OF_DAYS):
        for shift in range(SHIFTS_PER_DAY):
            if nurses_on_shift[day][shift] < REQUIRED_NURSES_PER_SHIFT:
                understaff_penalty += (REQUIRED_NURSES_PER_SHIFT - nurses_on_shift[day][shift]) * UNDER_STAFF_PENALTY
            elif nurses_on_shift[day][shift] > REQUIRED_NURSES_PER_SHIFT:
                overstaff_penalty += (nurses_on_shift[day][shift] - REQUIRED_NURSES_PER_SHIFT) * OVER_STAFF_PENALTY

    consecutive_shift_penalty = 0
    for nurse in range(NUMBER_OF_NURSES):
        consecutive_shifts = 0
        for day in range(NUMBER_OF_DAYS):
            if schedule[nurse][day] != -1:
                consecutive_shifts += 1
                if consecutive_shifts > MAX_CONSECUTIVE_SHIFTS:
                    consecutive_shift_penalty += CONSECUTIVE_SHIFT_PENALTY
            else:
                consecutive_shifts = 0

    preference_penalty = 0
    for nurse in range(NUMBER_OF_NURSES):
        for day in range(NUMBER_OF_DAYS):
            if schedule[nurse][day] != nurse_preferences[nurse][day]:
                preference_penalty += abs(schedule[nurse][day] - nurse_preferences[nurse][day]) * PREFERENCE_PENALTY

    total_penalty = (under_staffing_w1 * understaff_penalty) \
                    + (over_staffing_w2 * overstaff_penalty) \
                    + (consecutive_shift_w3 * consecutive_shift_penalty) \
                    + (preference_violation_w4 * preference_penalty)

    return (total_penalty,)


def genetic_algorithm():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, SHIFTS_PER_DAY - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int,
                     n=NUMBER_OF_NURSES * NUMBER_OF_DAYS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_schedule)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=SHIFTS_PER_DAY - 1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POPULATION_SIZE)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=GENERATIONS, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    best_schedule = np.reshape(best_individual, (NUMBER_OF_NURSES, NUMBER_OF_DAYS))
    return best_schedule, evaluate_schedule(best_individual)


def simulated_annealing():
    current_solution = np.random.randint(0, SHIFTS_PER_DAY, (NUMBER_OF_NURSES, NUMBER_OF_DAYS)).flatten()
    current_energy = evaluate_schedule(current_solution)[0]
    best_solution = current_solution
    best_energy = current_energy

    temperature = SA_INITIAL_TEMP
    cooling_rate = SA_COOLING_RATE
    iterations = ITERATIONS

    for i in range(iterations):
        temperature *= cooling_rate
        new_solution = current_solution.copy()
        nurse = random.randint(0, NUMBER_OF_NURSES - 1)
        day = random.randint(0, NUMBER_OF_DAYS - 1)
        new_solution[nurse * NUMBER_OF_DAYS + day] = random.randint(0, SHIFTS_PER_DAY - 1)

        new_energy = evaluate_schedule(new_solution)[0]
        if new_energy < current_energy or random.random() < np.exp((current_energy - new_energy) / temperature):
            current_solution = new_solution
            current_energy = new_energy

            if current_energy < best_energy:
                best_solution = current_solution
                best_energy = current_energy

    best_schedule = np.reshape(best_solution, (NUMBER_OF_NURSES, NUMBER_OF_DAYS))
    return best_schedule, best_energy


def ant_colony_optimization():
    num_ants = NUMBER_OF_ANTS
    iterations = ITERATIONS
    evaporation_rate = EVAPORATION_RATE
    alpha = ALPHA
    beta = BETA

    pheromone = np.ones((NUMBER_OF_NURSES, NUMBER_OF_DAYS, SHIFTS_PER_DAY))
    best_solution = None
    best_energy = float('inf')

    for iteration in range(iterations):
        solutions = []
        for ant in range(num_ants):
            solution = np.zeros((NUMBER_OF_NURSES, NUMBER_OF_DAYS), dtype=int)
            for nurse in range(NUMBER_OF_NURSES):
                for day in range(NUMBER_OF_DAYS):
                    probabilities = pheromone[nurse][day] ** alpha / np.sum(pheromone[nurse][day] ** alpha)
                    shift = np.random.choice(range(SHIFTS_PER_DAY), p=probabilities)
                    solution[nurse][day] = shift
            solutions.append(solution)

        for solution in solutions:
            energy = evaluate_schedule(solution.flatten())[0]
            if energy < best_energy:
                best_solution = solution
                best_energy = energy

        pheromone *= evaporation_rate
        for solution in solutions:
            for nurse in range(NUMBER_OF_NURSES):
                for day in range(NUMBER_OF_DAYS):
                    pheromone[nurse][day][solution[nurse][day]] += 1 / evaluate_schedule(solution.flatten())[0]

    return best_solution, best_energy


def particle_swarm_optimization():
    num_particles = NUMBER_OF_PARTICLES
    iterations = ITERATIONS
    w = 0.5
    c1 = 1.5
    c2 = 1.5

    particles = np.random.randint(0, SHIFTS_PER_DAY, (num_particles, NUMBER_OF_NURSES, NUMBER_OF_DAYS))
    velocities = np.zeros((num_particles, NUMBER_OF_NURSES, NUMBER_OF_DAYS))
    personal_best = particles.copy()
    personal_best_energies = [evaluate_schedule(p.flatten())[0] for p in personal_best]
    global_best = personal_best[np.argmin(personal_best_energies)]
    global_best_energy = min(personal_best_energies)

    for iteration in range(iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best[i] - particles[i]) +
                             c2 * r2 * (global_best - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], 0, SHIFTS_PER_DAY - 1)

            energy = evaluate_schedule(particles[i].flatten())[0]
            if energy < personal_best_energies[i]:
                personal_best[i] = particles[i]
                personal_best_energies[i] = energy
                if energy < global_best_energy:
                    global_best = particles[i]
                    global_best_energy = energy

    return global_best, global_best_energy


def tabu_search():
    tabu_size = TABU_SIZE
    iterations = ITERATIONS

    current_solution = np.random.randint(0, SHIFTS_PER_DAY, (NUMBER_OF_NURSES, NUMBER_OF_DAYS)).flatten()
    best_solution = current_solution
    best_energy = evaluate_schedule(current_solution)[0]
    tabu_list = []

    for iteration in range(iterations):
        neighbors = []
        for nurse in range(NUMBER_OF_NURSES):
            for day in range(NUMBER_OF_DAYS):
                for shift in range(SHIFTS_PER_DAY):
                    if shift != current_solution[nurse * NUMBER_OF_DAYS + day]:
                        neighbor = current_solution.copy()
                        neighbor[nurse * NUMBER_OF_DAYS + day] = shift
                        neighbors.append(neighbor)

        neighbors = [n for n in neighbors if not any(np.array_equal(n, t) for t in tabu_list)]
        if not neighbors:
            break

        current_solution = min(neighbors, key=lambda x: evaluate_schedule(x)[0])
        current_energy = evaluate_schedule(current_solution)[0]

        if current_energy < best_energy:
            best_solution = current_solution
            best_energy = current_energy

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    best_schedule = np.reshape(best_solution, (NUMBER_OF_NURSES, NUMBER_OF_DAYS))
    return best_schedule, best_energy


def main():
    print("Running Genetic Algorithm...")
    ga_schedule, ga_energy = genetic_algorithm()
    print(f"GA Best Schedule:\n{ga_schedule}\nGA Energy: {ga_energy}\n")

    print("Running Simulated Annealing...")
    sa_schedule, sa_energy = simulated_annealing()
    print(f"SA Best Schedule:\n{sa_schedule}\nSA Energy: {sa_energy}\n")

    print("Running Ant Colony Optimization...")
    aco_schedule, aco_energy = ant_colony_optimization()
    print(f"ACO Best Schedule:\n{aco_schedule}\nACO Energy: {aco_energy}\n")

    print("Running Particle Swarm Optimization...")
    pso_schedule, pso_energy = particle_swarm_optimization()
    print(f"PSO Best Schedule:\n{pso_schedule}\nPSO Energy: {pso_energy}\n")

    print("Running Tabu Search...")
    ts_schedule, ts_energy = tabu_search()
    print(f"TS Best Schedule:\n{ts_schedule}\nTS Energy: {ts_energy}\n")


if __name__ == "__main__":
    main()
