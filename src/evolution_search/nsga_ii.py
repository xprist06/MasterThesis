import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from evolution_search.individual import Individual
from evolution_search.population import Population
from evolution_search.nsgaii_utils import NSGAIIUtils
from model_engine.dataset import Dataset
from model_engine.model_utils import ModelUtils
from model_engine import layers
from tensorflow.keras.layers import Input
from copy import deepcopy
import random
import os
import datetime
import time
import uuid
import logging
import multiprocessing as mp


class NSGAII:
    def __init__(
            self,
            pop_size=10,
            max_gen=10,
            tournament_count=3,
            mutation_probability=0.1,
            phase_count=2,
            modules_count=4,
            genes_cros=True,
            dataset=0,
            batch_size=128,
            epochs=10,
            val_batch_size=64,
            val_epochs=25,
            val_split=0.1,
            verbose=2,
            datagen=None,
            result_export=None
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen

        if tournament_count <= pop_size:
            self.tournament_count = tournament_count
        else:
            self.tournament_count = pop_size

        if self.tournament_count == 0:
            self.roulette = True
        else:
            self.roulette = False

        self.mutation_probability = mutation_probability
        self.phase_count = phase_count
        self.modules_count = modules_count
        self.phase_output_idx = modules_count
        self.phase_skip_idx = modules_count + 1
        self.genes_cros = genes_cros
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_batch_size = val_batch_size
        self.val_epochs = val_epochs
        self.population = Population()
        self.dataset = Dataset(dataset)
        self.val_split = val_split
        self.verbose = verbose
        self.datagen = datagen

        if self.datagen is not None:
            self.datagen.fit(self.dataset.train_x)

        self.genotypes = []
        self.best_pf = None
        self.result_export = result_export
        self.output_log = ""

        NSGAIIUtils.phase_count = self.phase_count
        NSGAIIUtils.modules_count = self.modules_count
        NSGAIIUtils.phase_output_idx = self.modules_count
        NSGAIIUtils.phase_skip_idx = self.modules_count + 1

        dir_uid = uuid.uuid4()
        self.directory = "./" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(dir_uid)
        os.mkdir(self.directory)

        NSGAIIUtils.directory = self.directory
        self.utils = NSGAIIUtils()

        ModelUtils.input = Input(shape=(self.dataset.height, self.dataset.width, self.dataset.depth))
        ModelUtils.classes_cnt = self.dataset.classes_cnt
        ModelUtils.train_x = self.dataset.train_x
        ModelUtils.train_y = self.dataset.train_y
        ModelUtils.evo_train_x = self.dataset.evo_train_x
        ModelUtils.evo_train_y = self.dataset.evo_train_y
        ModelUtils.test_x = self.dataset.test_x
        ModelUtils.test_y = self.dataset.test_y
        ModelUtils.val_split = self.val_split
        ModelUtils.verbose = self.verbose
        ModelUtils.datagen = self.datagen
        ModelUtils.epochs = self.epochs
        ModelUtils.batch_size = self.batch_size
        ModelUtils.phase_count = self.phase_count
        ModelUtils.modules_count = self.modules_count
        ModelUtils.phase_output_idx = self.modules_count
        ModelUtils.phase_skip_idx = self.modules_count + 1

        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = True
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        ModelUtils.config = config
        self.model_utils = ModelUtils()

        start = time.time()

        # Evolution
        self.evolution()

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        logging.info("Computation time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        with open(self.directory + "/output.log", "a") as f:
            f.write("\nComputation time: ")
            f.write("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        # Save result
        self.save_result()

    def evolution(self):
        logging.info("Starting evolution...")
        self.write_log_prefix()
        if self.result_export.pareto_graph:
            self.utils.mk_pareto_dir()
        self.exploration()
        self.exploitation()
        # self.train_best_pf()
        self.set_best_pf()

        best_pf_output = "Best pareto front:\n"
        best_pf_output += self.utils.print_best_pf(self.best_pf)
        logging.info(best_pf_output)

        if self.result_export.pareto_graph:
            self.utils.make_pf_graph(self.population, self.best_pf, "val")

        with open(self.directory + "/output.log", "a") as f:
            f.write("                         AFTER VALIDATION\n\n")
        self.utils.log_best_pf(self.best_pf)

    def exploration(self):
        logging.info("Starting exploration phase...")
        self.initialize_population()
        # self.evaluate_population(self.population)
        self.utils.fast_nondominated_sort(self.population)
        for pareto_front in self.population.pareto_fronts:
            self.utils.compute_crowding_distance(pareto_front)
        self.population.compute_fitness()
        for i in range(self.max_gen):
            offsprings = self.create_offspring_population()
            # self.evaluate_population(offsprings)
            self.population.extend(offsprings.individuals)
            self.set_best_pf()
            self.population = self.create_new_population()

            best_pf_output = "Generation " + str(i + 1) + "/" + str(self.max_gen) + ":\n"
            best_pf_output += self.utils.print_best_pf(self.best_pf)
            logging.info(best_pf_output)

            if self.result_export.export:
                self.utils.save_best_pf(self.population, self.best_pf, self.result_export, str(i + 1))

            with open(self.directory + "/output.log", "a") as f:
                f.write("                             GENERATION " + str(i + 1) + "/" + str(self.max_gen) + "\n\n")
            self.utils.log_best_pf(self.best_pf)
            self.utils.log_population(self.population)

        print()
        print()

    def exploitation(self):
        logging.info("Starting exploitation phase...")
        if self.max_gen > 2:
            exp_gen = self.max_gen // 2
            if self.max_gen % 2 == 1:
                exp_gen += 1
        else:
            exp_gen = 2
        for i in range(exp_gen):
            offsprings = self.create_exploitation_offsprings()
            # self.evaluate_population(offsprings)
            self.population.extend(offsprings.individuals)
            self.set_best_pf()
            self.population = self.create_new_population()

            best_pf_output = "Exploitation " + str(i + 1) + "/" + str(exp_gen) + ":\n"
            best_pf_output += self.utils.print_best_pf(self.best_pf)
            logging.info(best_pf_output)

            if self.result_export.export:
                self.utils.save_best_pf(self.population, self.best_pf, self.result_export, "exp_" + str(i + 1))

            with open(self.directory + "/output.log", "a") as f:
                f.write("                           EXPLOITATION " + str(i + 1) + "/" + str(exp_gen) + "\n\n")
            self.utils.log_best_pf(self.best_pf)
            self.utils.log_population(self.population)

        print()
        print()

    def evaluate_population(self, population):
        for individual in population:
            manager = mp.Manager()
            queue = manager.Queue()
            p = mp.Process(target=ModelUtils.evaluate, args=(queue, individual))
            p.start()
            acc, error, params = queue.get()
            p.join()
            individual.accuracy = acc
            individual.error = error
            individual.param_count = params

    def train_best_pf(self):
        ModelUtils.batch_size = self.val_batch_size
        ModelUtils.epochs = self.val_epochs

        logging.info("Validation of best pareto front...")
        for individual in self.best_pf:
            manager = mp.Manager()
            queue = manager.Queue()
            p = mp.Process(target=ModelUtils.evaluate, args=(queue, individual, True))
            p.start()
            acc, error, params = queue.get()
            p.join()
            individual.accuracy = acc
            individual.error = error
            individual.param_count = params

    def set_best_pf(self):
        self.utils.fast_nondominated_sort(self.population)
        for pareto_front in self.population.pareto_fronts:
            self.utils.compute_crowding_distance(pareto_front)
        self.population.compute_fitness()
        if len(self.population.pareto_fronts[0]) > self.pop_size:
            self.population.pareto_fronts[0].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            self.best_pf = self.population.pareto_fronts[0][0:self.pop_size].copy()
        else:
            self.best_pf = self.population.pareto_fronts[0].copy()
        self.best_pf.sort(key=lambda individual: individual.error, reverse=False)

    def initialize_population(self):
        while len(self.population) < self.pop_size:
            new_individual = self.utils.generate_individual()
            if self.population.contains(new_individual):
                continue
            self.population.append(new_individual)

        for individual in self.population:
            self.genotypes.append(individual.genotype_to_str())

    def create_new_population(self):
        new_population = Population()
        last_pf = None
        for pareto_front in self.population.pareto_fronts:
            if len(new_population) + len(pareto_front) <= self.pop_size:
                new_population.extend(pareto_front)
            else:
                last_pf = pareto_front
                break
        last_pf.sort(key=lambda individual: individual.crowding_distance, reverse=True)
        new_population.extend(last_pf[0:self.pop_size - len(new_population)])
        new_population.compute_fitness()
        return new_population

    def create_offspring_population(self):
        offsprings = Population()
        while len(offsprings) < self.pop_size:
            parents = self.selection()
            if self.genes_cros:
                genotypes = self.utils.connections_crossover(parents)
            else:
                genotypes = self.utils.modules_crossover(parents)
            new_offsprings = []
            unique_offs = True
            for genotype in genotypes:
                if random.random() <= self.mutation_probability:
                    self.utils.mutation(genotype)
                else:
                    self.utils.minify_genotype(genotype)
                    self.utils.validate_genotype(genotype)
                new_individual = Individual(genotype)
                if self.genotype_exists(new_individual.genotype_to_str()):
                    unique_offs = False
                    break
                new_offsprings.append(new_individual)
            if unique_offs:
                offsprings.extend(new_offsprings)
                for offspring in new_offsprings:
                    self.genotypes.append(offspring.genotype_to_str())
            else:
                continue
        return offsprings

    def create_exploitation_offsprings(self):
        offsprings = Population()
        while len(offsprings) < self.pop_size:
            if self.roulette:
                parent = self.utils.roulette(self.population)[0]
            else:
                parent = self.utils.tournament(self.population, self.tournament_count)[0]
            genotype = deepcopy(parent.genotype)
            self.utils.mutation(genotype)
            new_individual = Individual(genotype)
            if self.genotype_exists(new_individual.genotype_to_str()):
                continue
            offsprings.append(new_individual)
            self.genotypes.append(new_individual.genotype_to_str())
        return offsprings

    def selection(self):
        if self.roulette:
            parents = self.utils.roulette(self.population)
        else:
            parents = self.utils.tournament(self.population, self.tournament_count)
        return parents

    def genotype_exists(self, new_genotype):
        for genotype in self.genotypes:
            if new_genotype == genotype:
                return True
        return False

    def write_log_prefix(self):
        with open(self.directory + "/output.log", "a") as f:
            f.write("-------------------------------------------------------------------\n")
            f.write("                          STARTUP SETTINGS                         \n")
            f.write("-------------------------------------------------------------------\n")
            f.write("\n")

            f.write("Dataset: ")

            if self.dataset.dataset == 0:
                f.write("MNIST\n")
            elif self.dataset.dataset == 1:
                f.write("Fashion MNIST\n")
            elif self.dataset.dataset == 2:
                f.write("SVHN\n")
            elif self.dataset.dataset == 3:
                f.write("CIFAR 10\n")
            elif self.dataset.dataset == 4:
                f.write("CIFAR 100\n")

            f.write("Population: ")
            f.write(str(self.pop_size) + "\n")
            f.write("Generations: ")
            f.write(str(self.max_gen) + "\n")
            f.write("Selection: ")
            if self.roulette:
                f.write("Roulette\n")
            else:
                f.write("Tournament\n")
                f.write("   Tournament count: ")
                f.write(str(self.tournament_count) + "\n")
            f.write("Mutation prob.: ")
            f.write(str(round(self.mutation_probability, 2)) + "\n")
            f.write("Phases: ")
            f.write(str(self.phase_count) + "\n")
            f.write("Modules: ")
            f.write(str(self.modules_count) + "\n")
            for i in range(self.modules_count):
                f.write("   " + str(i+1) + ": " + layers.phase_layers[i]["desc"] + "\n")
            f.write("Evolution batch size: ")
            f.write(str(self.batch_size) + "\n")
            f.write("Evolution epochs: ")
            f.write(str(self.epochs) + "\n")
            f.write("Validation batch size: ")
            f.write(str(self.val_batch_size) + "\n")
            f.write("Validation epochs: ")
            f.write(str(self.val_epochs) + "\n")
            f.write("Validation split: ")
            f.write(str(self.val_split) + "\n")
            f.write("Verbose: ")
            f.write(str(self.verbose) + "\n")

            if self.datagen is None:
                f.write("Data augmentation: False\n")
            else:
                f.write("Data augmentation: True\n")
                f.write("   Height shift: ")
                f.write(str(round(self.datagen.height_shift_range, 2)) + "\n")
                f.write("   Width shift: ")
                f.write(str(round(self.datagen.width_shift_range, 2)) + "\n")
                f.write("   Rotation range: ")
                f.write(str(self.datagen.rotation_range) + "\n")
                f.write("   Shear range: ")
                f.write(str(round(self.datagen.shear_range, 2)) + "\n")
                f.write("   Zoom range: ")
                f.write("[" + str(round(self.datagen.zoom_range[0], 2)) + "; " + str(round(self.datagen.zoom_range[1], 2)) + "]\n")
                f.write("   Horizontal flip: ")
                f.write(str(self.datagen.horizontal_flip) + "\n")
                f.write("   Vertical flip: ")
                f.write(str(self.datagen.vertical_flip) + "\n")
                f.write("   ZCA Whitening: ")
                f.write(str(self.datagen.zca_whitening) + "\n")
                f.write("   Featurewise center: ")
                f.write(str(self.datagen.featurewise_center) + "\n")
                f.write("   Samplewise center: ")
                f.write(str(self.datagen.samplewise_center) + "\n")
                f.write("   Featurewise std. norm.: ")
                f.write(str(self.datagen.featurewise_std_normalization) + "\n")
                f.write("   Samplewise std. norm.: ")
                f.write(str(self.datagen.samplewise_std_normalization) + "\n")

            if not self.result_export.export:
                f.write("Export: None\n")
            else:
                f.write("Export: \n")
                f.write("   Pareto graph: ")
                f.write(str(self.result_export.pareto_graph) + "\n")
                f.write("   Keras graph: ")
                f.write(str(self.result_export.keras_graph) + "\n")
                f.write("   Keras model: ")
                f.write(str(self.result_export.keras_model) + "\n")
                f.write("   Genotype graph/txt: ")
                f.write(str(self.result_export.genotype_info) + "\n")

            f.write("\n")
            f.write("-------------------------------------------------------------------\n")
            f.write("                             EVOLUTION                             \n")
            f.write("-------------------------------------------------------------------\n")
            f.write("\n")

    def save_result(self):
        logging.info("Saving result...")
        if self.result_export.export:
            self.utils.save_best_pf(
                self.population,
                self.best_pf,
                self.result_export
            )
        logging.info("Done")
