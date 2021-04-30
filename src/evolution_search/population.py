

class Population:
    def __init__(self):
        self.individuals = []
        self.population_rating = None
        self.pareto_fronts = None

    def __len__(self):
        return len(self.individuals)

    def __iter__(self):
        return self.individuals.__iter__()

    def contains(self, other):
        for idl in self.individuals:
            if idl == other:
                return True
        return False

    def extend(self, individuals):
        self.individuals.extend(individuals)

    def append(self, individual):
        self.individuals.append(individual)

    def compute_fitness(self):
        self.population_rating = 0.0
        for individual in self.individuals:
            individual.compute_fitness()
            self.population_rating += individual.fitness
