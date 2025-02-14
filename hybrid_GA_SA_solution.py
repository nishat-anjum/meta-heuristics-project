import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing

# Environment Parameters
NUMBER_OF_NURSES = 5
NUMBER_OF_DAYS = 7
SHIFTS_PER_DAY = 3

# Constraints
REQUIRED_NURSES_PER_SHIFT = 2
MAX_CONSECUTIVE_SHIFTS = 2
POPULATION_SIZE = 100

# Fitness Properties
UNDER_STAFF_PENALTY = 5
OVER_STAFF_PENALTY = 5
CONSECUTIVE_SHIFT_PENALTY = 8
PREFERENCE_PENALTY = 3

# penalty weights
under_staffing_w1 = 1
over_staffing_w2 = 1
consecutive_shift_w3 = 1.2
preference_violation_w4 = .5

# GA properties
GENERATIONS = 200

# SA properties
ITERATIONS = 200
SA_INITIAL_TEMP = 500
SA_COOLING_RATE = 0.995

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


def hybrid_ga_sa():
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
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=GENERATIONS, halloffame=hof, verbose=False)

    best_individuals = tools.selBest(population, k=5)

    with multiprocessing.Pool(processes=5) as pool:
        sa_results = pool.map(simulated_annealing, best_individuals)

    best_schedule, best_energy = min(sa_results, key=lambda x: x[1])

    return best_schedule, best_energy


def simulated_annealing(initial_solution):
    current_solution = np.array(initial_solution)
    current_energy = evaluate_schedule(current_solution)[0]
    best_solution = current_solution
    best_energy = current_energy

    temperature = SA_INITIAL_TEMP

    for _ in range(ITERATIONS):
        temperature *= SA_COOLING_RATE
        new_solution = current_solution.copy()

        nurse = random.randint(0, NUMBER_OF_NURSES - 1)
        day = random.randint(0, NUMBER_OF_DAYS - 1)
        new_solution[nurse * NUMBER_OF_DAYS + day] = random.randint(0, SHIFTS_PER_DAY - 1)

        new_energy = evaluate_schedule(new_solution)[0]
        delta = new_energy - current_energy

        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_solution = new_solution
            current_energy = new_energy
            if current_energy < best_energy:
                best_solution = current_solution
                best_energy = current_energy

    best_schedule = np.reshape(best_solution, (NUMBER_OF_NURSES, NUMBER_OF_DAYS))
    return best_schedule, best_energy


def print_schedule(schedule, energy):
    nurse_names = ["Nurse A", "Nurse B", "Nurse C", "Nurse D", "Nurse E"]
    shift_labels = {0: "Morning", 1: "Evening", 2: "Night"}
    print("\nOptimized Schedule:")
    for i, row in enumerate(schedule):
        shifts = [shift_labels[shift] for shift in row]
        print(f"{nurse_names[i]}: {shifts}")
    print(f"\nTotal Penalty Score: {energy}")
    print("Constraint Violations:")


def main():
    print("Running Hybrid GA + Simulated Annealing...")
    best_schedule, best_energy = hybrid_ga_sa()
    print_schedule(best_schedule, best_energy)
    print(f"Best Schedule:\n{best_schedule}\nBest Energy: {best_energy}\n")


if __name__ == "__main__":
    main()
