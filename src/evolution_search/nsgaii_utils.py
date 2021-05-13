# -----------------------------------------------------------------------------
# This software was developed as practical part of Master's thesis at FIT BUT
# The program uses multiobjective NSGA-II algorithm for designing accurate
# and compact CNNs.
#
# Author: Jan Pristas, xprist06@stud.fit.vutbr.cz
# Institute: Faculty of Information Technology, Brno University of Technology
#
# File: nsgaii_utils.py
# Description: Contains utility methods for evolution
# -----------------------------------------------------------------------------


from evolution_search.individual import Individual
from model_engine.model_utils import ModelUtils
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from graphviz import Digraph
import random
import math
import os
import imageio
import gc


class NSGAIIUtils:
    phase_count = None
    modules_count = None
    phase_output_idx = None
    phase_skip_idx = None
    directory = None

    ####################################################################################################################
    #                                             EVOLUTION METHODS                                                    #
    ####################################################################################################################

    def fast_nondominated_sort(self, population):
        """
        Sort population into Pareto fronts
        Method was taken over from Github repository
        - https://github.com/baopng/NSGA-II/blob/master/nsga2/utils.py
        :param population: Population
        """
        population.pareto_fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_individuals = []
            for another_individual in population:
                if individual is not another_individual:
                    if individual.dominates(another_individual):
                        individual.dominated_individuals.append(another_individual)
                    elif another_individual.dominates(individual):
                        individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.pareto_fronts[0].append(individual)
        i = 0
        while len(population.pareto_fronts[i]) > 0:
            tmp_pareto_front = []
            for individual in population.pareto_fronts[i]:
                for another_individual in individual.dominated_individuals:
                    another_individual.domination_count -= 1
                    if another_individual.domination_count == 0:
                        another_individual.rank = i + 1
                        tmp_pareto_front.append(another_individual)
            i += 1
            population.pareto_fronts.append(tmp_pareto_front)

    def compute_crowding_distance(self, pareto_front):
        """
        Compute crowding distance of individuals in Pareto front
        Method was taken over from Github repository
        - https://github.com/baopng/NSGA-II/blob/master/nsga2/utils.py
        :param pareto_front: Pareto front
        """
        if len(pareto_front) > 0:
            individuals_count = len(pareto_front)
            for individual in pareto_front:
                individual.crowding_distance = 0

            pareto_front.sort(key=lambda individual: individual.error)
            pareto_front[0].crowding_distance = math.inf
            pareto_front[individuals_count - 1].crowding_distance = math.inf
            obj_values = [individual.error for individual in pareto_front]
            scale = max(obj_values) - min(obj_values)
            if scale == 0:
                scale = 1
            for i in range(1, individuals_count - 1):
                pareto_front[i].crowding_distance += (pareto_front[i + 1].error - pareto_front[i - 1].error) / scale

            pareto_front.sort(key=lambda individual: individual.param_count)
            pareto_front[0].crowding_distance = math.inf
            pareto_front[individuals_count - 1].crowding_distance = math.inf
            obj_values = [individual.param_count for individual in pareto_front]
            scale = max(obj_values) - min(obj_values)
            if scale == 0:
                scale = 1
            for i in range(1, individuals_count - 1):
                pareto_front[i].crowding_distance += (pareto_front[i + 1].param_count - pareto_front[
                    i - 1].param_count) / scale

    def tournament(self, population, tournament_count):
        """
        Tournament selection
        :param population: Population
        :param tournament_count: Number of individuals randomly selected into tournament
        :return: New parents
        """
        parents = []
        for i in range(2):
            tmp_idls = random.sample(population.individuals, tournament_count)
            best_idl = tmp_idls[0]
            for idl in tmp_idls:
                if idl < best_idl:
                    best_idl = idl
            parents.append(best_idl)
        return parents

    def roulette(self, population):
        """
        Roulette selection
        :param population: Population
        :return: New parents
        """
        weights = []
        roulette_wheel = []
        for individual in population:
            weights.append(individual.fitness/population.population_rating)
        roulette_wheel.append(weights[0])
        for i in range(1, len(population)):
            roulette_wheel.append(roulette_wheel[i-1] + weights[i])
        parents = random.choices(population.individuals, cum_weights=roulette_wheel, k=2)
        return parents

    def connections_crossover(self, parents):
        """
        Create new individuals using connections (genes) crossover
        :param parents: Parents
        :return: New offsprings
        """
        offsprings = []
        offsprings_genotype_arr = []
        parent_genotype_arr = []
        for i in range(2):
            offsprings_genotype_arr.append([])
            parent_genotype_arr.append(self.genotype_to_arr(parents[i].genotype))
        length = len(parent_genotype_arr[0])
        for i in range(length):
            first_parent = random.randint(0, 1)
            if first_parent == 0:
                second_parent = 1
            else:
                second_parent = 0
            offsprings_genotype_arr[0].append(parent_genotype_arr[first_parent][i].copy())
            offsprings_genotype_arr[1].append(parent_genotype_arr[second_parent][i].copy())
        for genotype_arr in offsprings_genotype_arr:
            offsprings.append(self.arr_to_genotype(genotype_arr))
        return offsprings

    def genotype_to_arr(self, genotype):
        """
        Convert genotype into 1d array
        :param genotype: Genotype
        :return: 1d array representation of genotype
        """
        genotype_arr = []
        self.expand_genotype(genotype)
        for i in range(NSGAIIUtils.phase_count):
            for j in range(NSGAIIUtils.modules_count + 2):
                for k in range(len(genotype[i][j])):
                    genotype_arr.append(genotype[i][j][k].copy())
        self.minify_genotype(genotype)
        return genotype_arr

    def arr_to_genotype(self, genotype_arr):
        """
        Convert 1d array into genotype
        :param genotype_arr: 1d array
        :return: Genotype
        """
        genotype = []
        for i in range(NSGAIIUtils.phase_count):
            genotype.append([])
            for j in range(NSGAIIUtils.modules_count + 2):
                if j == NSGAIIUtils.phase_skip_idx:
                    genotype[i].append(np.array(genotype_arr[0:1].copy()))
                    del genotype_arr[0]
                elif j == NSGAIIUtils.phase_output_idx:
                    genotype[i].append(np.array(genotype_arr[0:j].copy()))
                    del genotype_arr[0:j]
                else:
                    genotype[i].append(np.array(genotype_arr[0:j+1].copy()))
                    del genotype_arr[0:j+1]
        self.minify_genotype(genotype)
        return genotype

    def modules_crossover(self, parents):
        """
        Create new individuals using modules crossover
        :param parents: Parents
        :return: New offsprings
        """
        offsprings = []
        for i in range(2):
            offsprings.append([])
        for i in range(NSGAIIUtils.phase_count):
            offsprings[0].append([])
            offsprings[1].append([])
            for j in range(NSGAIIUtils.modules_count + 2):
                first_parent = random.randint(0, 1)
                if first_parent == 0:
                    second_parent = 1
                else:
                    second_parent = 0
                offsprings[0][i].append(parents[first_parent].genotype[i][j].copy())
                offsprings[1][i].append(parents[second_parent].genotype[i][j].copy())
            offsprings[0][i][NSGAIIUtils.phase_output_idx] = np.zeros(NSGAIIUtils.modules_count, dtype=int)
            offsprings[1][i][NSGAIIUtils.phase_output_idx] = np.zeros(NSGAIIUtils.modules_count, dtype=int)
        return offsprings

    def mutation(self, genotype):
        """
        Mutate individual's genotype
        :param genotype: genotype
        :return: Mutated genotype
        """
        phase = random.randint(0, NSGAIIUtils.phase_count - 1)
        module = random.randint(0, NSGAIIUtils.modules_count + 1)
        while module == NSGAIIUtils.phase_output_idx:
            module = random.randint(0, NSGAIIUtils.modules_count + 1)
        if module == NSGAIIUtils.phase_skip_idx:
            if genotype[phase][module][0] == 0:
                genotype[phase][module][0] = 1
            else:
                genotype[phase][module][0] = 0
        elif genotype[phase][module][0] == 0:
            genotype[phase][module] = np.zeros(module + 1, dtype=int)
            genotype[phase][module][0] = 1
        else:
            connection = random.randint(0, module)
            if genotype[phase][module][connection] == 0:
                genotype[phase][module][connection] = 1
            else:
                genotype[phase][module][connection] = 0
        self.minify_genotype(genotype)
        self.validate_genotype(genotype)

    def generate_individual(self):
        """
        Generate new individual
        :return: New individual
        """
        genotype = []
        for i in range(NSGAIIUtils.phase_count):
            genotype.append([])
            for j in range(NSGAIIUtils.modules_count + 2):
                if j == NSGAIIUtils.phase_skip_idx:
                    genotype[i].append(np.random.randint(2, size=1))
                elif j == NSGAIIUtils.phase_output_idx:
                    genotype[i].append(np.zeros(NSGAIIUtils.modules_count, dtype=int))
                else:
                    genotype[i].append(np.random.randint(2, size=j + 1))
        self.minify_genotype(genotype)
        self.validate_genotype(genotype)
        individual = Individual(genotype)
        return individual

    def minify_genotype(self, genotype):
        """
        Minify genotype
        :param genotype: Genotype
        """
        for i in range(NSGAIIUtils.phase_count):
            for j in range(NSGAIIUtils.modules_count):
                if genotype[i][j][0] == 0 and len(genotype[i][j]) > 1:
                    genotype[i][j] = np.array([0])

    def expand_genotype(self, genotype):
        """
        Expand genotype
        :param genotype: Genotype
        """
        for i in range(NSGAIIUtils.phase_count):
            for j in range(NSGAIIUtils.modules_count):
                if j != 0:
                    if genotype[i][j][0] == 0:
                        genotype[i][j] = np.zeros(j + 1, dtype=int)

    def validate_genotype(self, genotype):
        """
        Validate genotype - fix invalid connections
        :param genotype: Genotype
        """
        for i in range(NSGAIIUtils.phase_count):
            for j in range(NSGAIIUtils.modules_count):
                if genotype[i][j][0] == 1:
                    is_out_used = False
                    for k in range(j + 1, NSGAIIUtils.modules_count):
                        if genotype[i][k][0] == 1:
                            if genotype[i][k][j + 1] == 1:
                                is_out_used = True
                                break
                    if not is_out_used:
                        genotype[i][NSGAIIUtils.phase_output_idx][j] = 1
                else:
                    for k in range(j + 1, NSGAIIUtils.modules_count):
                        if genotype[i][k][0] == 1:
                            if genotype[i][k][j + 1] == 1:
                                genotype[i][k][j + 1] = 0
                    if genotype[i][NSGAIIUtils.phase_output_idx][j] == 1:
                        genotype[i][NSGAIIUtils.phase_output_idx][j] = 0

    ####################################################################################################################
    #                                                  LOGGER METHODS                                                  #
    ####################################################################################################################

    def log_best_pf(self, best_pf):
        """
        Log best Pareto front informations
        :param best_pf: Best Pareto front
        """
        i = 0
        with open(self.directory + "/output.log", "a") as f:
            f.write("BEST PARETO FRONT:\n")
            for individual in best_pf:
                f.write(str(i) + ": error: " + str(individual.error) + "   param_count: " + str(individual.param_count) + "\n")
                i += 1
            f.write("\n-------------------------------------------------------------------\n\n")

    def log_population(self, population):
        """
        Log population informations
        :param population: Population
        """
        i = 0
        with open(self.directory + "/output.log", "a") as f:
            f.write("POPULATION:\n")
            for individual in population:
                f.write("Individual " + str(individual.id) + ": \n")
                f.write("   Accuracy: " + str(individual.accuracy) + "\n")
                f.write("   Error: " + str(individual.error) + "\n")
                f.write("   Param. count: " + str(individual.param_count) + "\n")
                f.write("   Genotype: \n")
                f.write("      " + individual.genotype_to_str() + "\n\n")
                i += 1
            f.write("-------------------------------------------------------------------\n")
            f.write("-------------------------------------------------------------------\n\n")

    ####################################################################################################################
    #                                                  PARETO METHODS                                                  #
    ####################################################################################################################

    def mk_pareto_dir(self):
        """
        Make tmp_pareto directory
        If already exists, remove and create empty one
        """
        if os.path.isdir(NSGAIIUtils.directory + "/tmp_pareto"):
            self.rm_pareto_dir()
        directory = NSGAIIUtils.directory + "/tmp_pareto"
        os.mkdir(directory)

    def rm_pareto_dir(self):
        """
        Remove tmp_pareto directory
        """
        directory = NSGAIIUtils.directory + "/tmp_pareto"
        if len(os.listdir(directory)) == 0:
            os.rmdir(directory)
        else:
            files = os.listdir(directory)
            for file in files:
                os.remove(directory + "/" + file)
            os.rmdir(directory)

    def print_best_pf(self, best_pf):
        """
        Create info string about best pareto front for output log
        :param best_pf: Best Pareto front
        :return: String with info about best Pareto front
        """
        i = 0
        output = ""
        for individual in best_pf:
            output += str(i) + ": error: " + str(individual.error) + "\tparam_count: " + str(individual.param_count) + "\n"
            i += 1
        return output

    def make_pf_graph(self, population, best_pf, idx=None):
        """
        Make population graph with highlighted best Pareto front
        :param population: Population
        :param best_pf: Best Pareto front
        :param idx: Index of generation, None after computation for last Pareto front graph
        """
        scores = np.empty((0, 2), float)
        pareto_scores = np.empty((0, 2), float)

        for individual in population:
            score = np.array([[individual.param_count, round(individual.error, 3)]])
            scores = np.append(scores, score, axis=0)

        for individual in best_pf:
            score = np.array([[individual.param_count, round(individual.error, 3)]])
            pareto_scores = np.append(pareto_scores, score, axis=0)

        x = scores[:, 0]
        y = scores[:, 1]
        x_pareto = pareto_scores[:, 0]
        y_pareto = pareto_scores[:, 1]

        plt.scatter(x, y)
        plt.plot(x_pareto, y_pareto, 'o-r', color='r')
        plt.xlabel('Parameters count')
        plt.ylabel('Validation error (%)')
        if idx is None:
            directory = NSGAIIUtils.directory + "/result"
            os.mkdir(directory)
            plt.savefig(directory + "/pareto_graph.pdf")
        elif idx.startswith("exp_"):
            plt.title("Exploitation " + idx[4:])
            plt.savefig(NSGAIIUtils.directory + "/tmp_pareto" + "/exp_" + idx[4:] + ".png")
            directory = NSGAIIUtils.directory + "/exp_" + idx[4:]
            os.mkdir(directory)
            plt.savefig(directory + "/pareto_graph.pdf")
        elif idx != "val":
            plt.title("Generation " + idx)
            plt.savefig(NSGAIIUtils.directory + "/tmp_pareto" + "/gen_" + idx + ".png")
            directory = NSGAIIUtils.directory + "/gen_" + idx
            os.mkdir(directory)
            plt.savefig(directory + "/pareto_graph.pdf")
        else:
            plt.title("After validation")
            plt.savefig(NSGAIIUtils.directory + "/tmp_pareto" + "/val.png")
        plt.clf()

    def make_pf_gif(self):
        """
        Make Pareto front evolution gif from graphs in tmp_pareto directory
        """
        images = []
        path = NSGAIIUtils.directory + "/tmp_pareto"
        files = sorted(os.listdir(path), key=lambda x: os.path.getctime(os.path.join(path, x)))
        for file in files:
            if file.endswith(".png"):
                images.append(imageio.imread(path + "/" + file))
        imageio.mimsave(NSGAIIUtils.directory + "/result" + "/pareto_evolution.gif", images, fps=1)
        self.rm_pareto_dir()

    def save_best_pf(self, population, best_pf, result_export=None, idx=None):
        """
        Save informations about best Pareto front
        :param population: Population
        :param best_pf: Best Pareto front
        :param result_export: Result export info
        :param idx: Index of generation
        """
        if result_export is not None:
            if result_export.pareto_graph:
                self.make_pf_graph(population, best_pf, idx)
                if idx is None:
                    self.make_pf_gif()
            if result_export.export_individual():
                res = False
                if idx is None:
                    directory = NSGAIIUtils.directory + "/result"
                    res = True
                elif idx.startswith("exp_"):
                    directory = NSGAIIUtils.directory + "/exp_" + idx[4:]
                else:
                    directory = NSGAIIUtils.directory + "/gen_" + idx
                if not os.path.isdir(directory):
                    os.mkdir(directory)

                for individual in best_pf:
                    if idx is None:
                        individual_directory = directory + "/individual_" + str(individual.id)
                    elif idx.startswith("exp_"):
                        individual_directory = directory + "/individual_" + str(individual.id)
                    else:
                        individual_directory = directory + "/individual_" + str(individual.id)
                    os.mkdir(individual_directory)
                    self.write_ind_to_file(
                        individual,
                        individual_directory,
                        result_export,
                        res=res
                    )

    ####################################################################################################################
    #                                                INDIVIDUAL METHODS                                                #
    ####################################################################################################################

    def make_graph(self, individual):
        """
        Make genotype graph using Digraph
        Implementation of method was inspired in project from Github repository
        - https://github.com/ianwhale/nsga-net/blob/master/visualization/macro_visualize.py
        :param individual: Individual
        :return: Individual's genotype graph
        """
        conv_color = "orangered"
        node_color = "lightblue"
        input_color = "white"
        sum_color = "limegreen"
        pool_color = "orange"
        drop_color = "darkorchid1"
        phase_background_color = "snow3"
        output_color = "yellow1"

        node_shape = "circle"

        graph_structure = self.make_graph_structure(individual)

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')

        graph_attr = dict(size="12,12")

        graph = Digraph(format="pdf", filename="graph", node_attr=node_attr, graph_attr=graph_attr)
        graph.attr(rankdir="LR")
        graph.attr(label="Individual_" + str(individual.id) + "\n\n")
        graph.attr(labelloc='t')

        graph.node("input", "Input", fillcolor=input_color)
        graph.node("input_conv", " ", fillcolor=conv_color, shape=node_shape)
        graph.edge("input", "input_conv")

        nodes = graph_structure['nodes']
        pool = graph_structure['pool']
        drop = graph_structure['drop']

        for i in range(NSGAIIUtils.phase_count):
            graph.node(nodes[i][len(nodes[i])-1]['id'], nodes[i][len(nodes[i])-1]['label'],
                       fillcolor=sum_color, shape=node_shape)
            graph.node(pool[i]['id'], pool[i]['label'], fillcolor=pool_color)
            graph.node(drop[i]['id'], drop[i]['label'], fillcolor=drop_color)
            with graph.subgraph(name="cluster_" + str(i)) as p:
                p.attr(fillcolor=phase_background_color, label='', fontcolor="black", style="filled")
                for j in range(len(nodes[i]) - 1):
                    node_id = nodes[i][j]['id']
                    node_label = nodes[i][j]['label']
                    p.node(node_id, node_label, fillcolor=node_color, shape=node_shape)

        for edge in graph_structure['edges']:
            graph.edge(edge['source'], edge['sink'])

        graph.node("linear", "Linear", fillcolor=output_color)
        graph.edge(drop[-1]['id'], 'linear')

        return graph

    def make_graph_structure(self, individual):
        """
        Make graph structure from genotype
        :param individual: Individual
        :return: Graph structure
        """
        nodes = []
        pool_nodes = []
        dropout_nodes = []
        edges = []

        for i in range(NSGAIIUtils.phase_count):
            nodes.append([])
            for j in range(NSGAIIUtils.modules_count):
                if individual.genotype[i][j][0] == 1:
                    nodes[i].append({
                        "id": "node_" + str(i) + "_" + str(j+1),
                        "label": str(j+1)
                    })
            nodes[i].append({
                "id": "node_" + str(i) + "_output",
                "label": " "
            })
            pool_nodes.append({
                "id": "pool_" + str(i),
                "label": "AvgPool"
            })
            dropout_nodes.append({
                "id": "drop_" + str(i),
                "label": "Dropout"
            })

        for i in range(NSGAIIUtils.phase_count):
            if individual.phases[i].is_empty():
                if i > 0:
                    edges.append({
                        "source": "drop_" + str(i-1),
                        "sink": "node_" + str(i) + "_output"
                    })
                else:
                    edges.append({
                        "source": "input_conv",
                        "sink": "node_" + str(i) + "_output"
                    })
            else:
                for j in range(NSGAIIUtils.modules_count):
                    if individual.genotype[i][j][0] == 1:
                        phase_input_used = True
                        for k in range(1, j+1):
                            if individual.genotype[i][j][k] == 1:
                                phase_input_used = False
                                edges.append({
                                    "source": "node_" + str(i) + "_" + str(k),
                                    "sink": "node_" + str(i) + "_" + str(j+1)
                                })
                        if phase_input_used:
                            if i > 0:
                                edges.append({
                                    "source": "drop_" + str(i-1),
                                    "sink": "node_" + str(i) + "_" + str(j+1)
                                })
                            else:
                                edges.append({
                                    "source": "input_conv",
                                    "sink": "node_" + str(i) + "_" + str(j+1)
                                })

            if individual.genotype[i][NSGAIIUtils.phase_skip_idx][0] == 1:
                if i > 0:
                    edges.append({
                        "source": "drop_" + str(i-1),
                        "sink": "node_" + str(i) + "_output"
                    })
                else:
                    edges.append({
                        "source": "input_conv",
                        "sink": "node_" + str(i) + "_output"
                    })
            for j in range(NSGAIIUtils.modules_count):
                if individual.genotype[i][NSGAIIUtils.phase_output_idx][j] == 1:
                    edges.append({
                        "source": "node_" + str(i) + "_" + str(j+1),
                        "sink": "node_" + str(i) + "_output"
                    })
            edges.append({
                "source": "node_" + str(i) + "_output",
                "sink": "pool_" + str(i)
            })

            edges.append({
                "source": "pool_" + str(i),
                "sink": "drop_" + str(i)
            })

        graph_structure = {
            "nodes": nodes,
            "pool": pool_nodes,
            "drop": dropout_nodes,
            "edges": edges
        }

        return graph_structure

    def write_ind_to_file(self, individual, directory, result_export, res=False):
        """
        Write individual's informations into files
        :param individual: Individual
        :param directory: Individual's directory
        :param result_export: Result export info
        :param res: Flag if is computation over, if True, individual is one of result
        """
        if res:
            if result_export.export_keras():
                try:
                    model = ModelUtils.create_model(individual)
                    if result_export.keras_graph:
                        keras.utils.plot_model(model, to_file=directory + "/cnn_structure.png", show_shapes=True)
                    if result_export.keras_model:
                        model_json = model.to_json()
                        with open(directory + "/model.json", "w") as f:
                            f.write(model_json)
                        del model_json
                    del model
                except tf.errors.ResourceExhaustedError as e:
                    print("[INFO] Individual " + str(individual.id) + "'s model can't fit into memory. Skipping...")
                finally:
                    gc.collect()

        if result_export.genotype_info:
            graph = self.make_graph(individual)
            graph.render("genotype_graph", directory)
            with open(directory + "/genotype.txt", "w") as f:
                f.write(individual.__str__())
