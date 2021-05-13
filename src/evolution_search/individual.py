# -----------------------------------------------------------------------------
# This software was developed as practical part of Master's thesis at FIT BUT
# The program uses multiobjective NSGA-II algorithm for designing accurate
# and compact CNNs.
#
# Author: Jan Pristas, xprist06@stud.fit.vutbr.cz
# Institute: Faculty of Information Technology, Brno University of Technology
#
# File: individual.py
# Description: Contains Individual class
# -----------------------------------------------------------------------------


from model_engine.phase import Phase
import math


class Individual:
    id = 0

    def __init__(self, genotype):
        self.id = Individual.id
        Individual.id += 1
        self.phases = []
        self.fitness = None
        self.accuracy = None
        self.error = None
        self.param_count = None
        self.genotype = genotype
        self.domination_count = 0
        self.dominated_individuals = []
        self.rank = 0
        self.crowding_distance = 0
        self.training_time = None

        for phase in self.genotype:
            self.phases.append(Phase(phase))

    def __str__(self):
        output = ""
        output += "Accuracy: " + str(self.accuracy) + "\n"
        output += "Error: " + str(self.error) + "\n"
        output += "Parameters count: " + str(self.param_count) + "\n"
        if self.training_time is not None:
            hours, rem = divmod(self.training_time, 3600)
            minutes, seconds = divmod(rem, 60)
            output += "Training time: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds)
        output += "Genotype:\n"
        output += self.genotype_to_str()
        return output

    def __eq__(self, other):
        if isinstance(other, Individual):
            return self.genotype_to_str() == other.genotype_to_str()
        return False

    def __lt__(self, other):
        if isinstance(other, Individual):
            return (self.rank < other.rank) \
                   or ((self.rank == other.rank)
                       and (self.crowding_distance > other.crowding_distance))

    def dominates(self, other):
        """
        Check if individual dominates other
        :param other: Other individual
        :return: True if individual dominates other, False otherwise
        """
        if isinstance(other, Individual):
            return (self.error <= other.error and self.param_count <= other.param_count) \
                   and (self.error < other.error or self.param_count < other.param_count)

    def compute_fitness(self):
        """
        Compute crowding distance of individual
        """
        if self.crowding_distance == math.inf or self.crowding_distance == 0:
            tmp_val = self.rank + 1
        else:
            tmp_val = self.rank + (1/self.crowding_distance) + 1
        self.fitness = 1/tmp_val

    def genotype_to_str(self):
        """
        Convert individual's genotype from array to string
        :return: String representation of individual's genotype
        """
        str_genotype = ""
        for phase in self.phases:
            for i in range(len(phase.phase_connections)):
                for j in range(len(phase.phase_connections[i])):
                    str_genotype += str(phase.phase_connections[i][j])
                str_genotype += "-"
            str_genotype = str_genotype[:-1]
            str_genotype += " --- "
        return str_genotype[:-5]
