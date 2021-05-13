# -----------------------------------------------------------------------------
# This software was developed as practical part of Master's thesis at FIT BUT
# The program uses multiobjective NSGA-II algorithm for designing accurate
# and compact CNNs.
#
# Author: Jan Pristas, xprist06@stud.fit.vutbr.cz
# Institute: Faculty of Information Technology, Brno University of Technology
#
# File: population.py
# Description: Contains Population class
# -----------------------------------------------------------------------------


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
        """
        Check if population contains individual
        :param other: Individual to be checked
        :return: Boolean value of individual occurrence in population
        """
        for idl in self.individuals:
            if idl == other:
                return True
        return False

    def extend(self, individuals):
        """
        Extend population of new individuals
        :param individuals: New individuals
        """
        self.individuals.extend(individuals)

    def append(self, individual):
        """
        Append new individual into population
        :param individual: New individual
        """
        self.individuals.append(individual)

    def compute_fitness(self):
        """
        Compute fitness of individuals and compute population rating
        """
        self.population_rating = 0.0
        for individual in self.individuals:
            individual.compute_fitness()
            self.population_rating += individual.fitness
