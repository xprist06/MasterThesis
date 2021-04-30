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
            genes_conn=True,
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
        self.genes_conn = genes_conn
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

        directory = "./" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.mkdir(directory)

        NSGAIIUtils.directory = directory

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
        print("[INFO] Computation time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        self.output_log += "\n"
        self.output_log += "Computation time: "
        self.output_log += "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

        # Save result
        self.save_result()

    def evolution(self):
        print("[INFO] Starting evolution...")
        self.write_log_prefix()
        if self.result_export.pareto_graph:
            self.utils.mk_pareto_dir()
        self.exploration()
        self.exploitation()
        self.train_best_pf()
        self.set_best_pf()
        self.utils.print_best_pf(self.best_pf)
        if self.result_export.pareto_graph:
            self.utils.make_pf_graph(self.population, self.best_pf, "val")
        self.output_log += "                         AFTER VALIDATION\n\n"
        self.output_log = self.utils.log_best_pf(self.output_log, self.best_pf)

    def exploration(self):
        print("[INFO] Starting exploration phase...")
        self.initialize_population()
        self.evaluate_population(self.population)
        self.utils.fast_nondominated_sort(self.population)
        for pareto_front in self.population.pareto_fronts:
            self.utils.compute_crowding_distance(pareto_front)
        self.population.compute_fitness()
        for i in range(self.max_gen):
            offsprings = self.create_offspring_population()
            self.evaluate_population(offsprings)
            self.population.extend(offsprings.individuals)
            self.set_best_pf()
            self.population = self.create_new_population()
            if self.result_export.pareto_graph:
                self.utils.make_pf_graph(self.population, self.best_pf, str(i + 1))
            print("[INFO] Generation " + str(i + 1) + "/" + str(self.max_gen) + ":")
            self.output_log += "                             GENERATION " + str(i + 1) + "/" + str(self.max_gen) + "\n\n"
            self.output_log = self.utils.log_best_pf(self.output_log, self.best_pf)
            self.output_log = self.utils.log_population(self.output_log, self.population)
            self.utils.print_best_pf(self.best_pf)
        print()

    def exploitation(self):
        print("[INFO] Starting exploitation phase...")
        if self.max_gen > 2:
            exp_gen = self.max_gen // 2
            if self.max_gen % 2 == 1:
                exp_gen += 1
        else:
            exp_gen = 2
        for i in range(exp_gen):
            offsprings = self.create_exploitation_offsprings()
            self.evaluate_population(offsprings)
            self.population.extend(offsprings.individuals)
            self.set_best_pf()
            self.population = self.create_new_population()
            if self.result_export.pareto_graph:
                self.utils.make_pf_graph(self.population, self.best_pf, "exp_" + str(i + 1))
            print("[INFO] Exploitation " + str(i + 1) + "/" + str(exp_gen) + ":")
            self.output_log += "                           EXPLOITATION " + str(i + 1) + "/" + str(exp_gen) + "\n\n"
            self.output_log = self.utils.log_best_pf(self.output_log, self.best_pf)
            self.output_log = self.utils.log_population(self.output_log, self.population)
            self.utils.print_best_pf(self.best_pf)
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

        print("[INFO] Validation of best pareto front...")
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
        # genotype = [
        #     [
        #         [1], [1, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1]
        #     ],
        #     [
        #         [1], [1, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1]
        #     ],
        #     [
        #         [1], [1, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1]
        #     ]
        # ]
        #
        # self.population.append(Individual(genotype))

        while len(self.population) < self.pop_size:
            new_individual = self.utils.generate_individual()
            if self.population.contains(new_individual):
                continue
            self.population.append(new_individual)

        for individual in self.population:
            self.genotypes.append(individual.genotype_to_str())

        # self.population.individuals[0].error = 16.360002756118774
        # self.population.individuals[0].param_count = 1888234
        # self.population.individuals[1].error = 15.10000228881836
        # self.population.individuals[1].param_count = 1626634
        # self.population.individuals[2].error = 15.270000696182251
        # self.population.individuals[2].param_count = 1513418
        # self.population.individuals[3].error = 15.57999849319458
        # self.population.individuals[3].param_count = 1451594
        # self.population.individuals[4].error = 17.500001192092896
        # self.population.individuals[4].param_count = 1144202
        # self.population.individuals[5].error = 19.739997386932373
        # self.population.individuals[5].param_count = 1097098
        # self.population.individuals[6].error = 18.300002813339233
        # self.population.individuals[6].param_count = 1050826
        # self.population.individuals[7].error = 19.209998846054077
        # self.population.individuals[7].param_count = 895658
        # self.population.individuals[8].error = 19.630002975463867
        # self.population.individuals[8].param_count = 855178
        # self.population.individuals[9].error = 20.20999789237976
        # self.population.individuals[9].param_count = 832362
        # self.population.individuals[10].error = 19.739997386932373
        # self.population.individuals[10].param_count = 689066
        # self.population.individuals[11].error = 19.47000026702881
        # self.population.individuals[11].param_count = 676874
        # self.population.individuals[12].error = 20.27999758720398
        # self.population.individuals[12].param_count = 545354
        # self.population.individuals[13].error = 21.729999780654907
        # self.population.individuals[13].param_count = 540138
        # self.population.individuals[14].error = 21.310001611709595
        # self.population.individuals[14].param_count = 464010
        # self.population.individuals[15].error = 25.520002841949463
        # self.population.individuals[15].param_count = 340042
        # self.population.individuals[16].error = 24.18000102043152
        # self.population.individuals[16].param_count = 332138
        # self.population.individuals[17].error = 27.160000801086426
        # self.population.individuals[17].param_count = 306090
        # self.population.individuals[18].error = 28.09000015258789
        # self.population.individuals[18].param_count = 241834
        # self.population.individuals[19].error = 27.99999713897705
        # self.population.individuals[19].param_count = 200234
        # self.population.individuals[20].error = 30.070000886917114
        # self.population.individuals[20].param_count = 179562
        # self.population.individuals[21].error = 31.88999891281128
        # self.population.individuals[21].param_count = 156650
        # self.population.individuals[22].error = 39.82999920845032
        # self.population.individuals[22].param_count = 135978
        #
        # for individual in self.population:
        #     individual.fitness = individual.error + individual.param_count
        #
        # self.utils.fast_nondominated_sort(self.population)
        # self.best_pf = self.population.pareto_fronts[0]
        # self.best_pf.sort(key=lambda individual: individual.fitness, reverse=True)
        # self.utils.save_best_pf(self.population, self.best_pf)

        # genotype_1 = [
        #     [
        #         [0], [1, 0], [1, 0, 0], [0], [0], [0], [0, 1, 1, 0, 0, 0], [0]
        #     ],
        #     [
        #         [0], [1, 0], [0], [1, 0, 0, 0], [0], [0], [0, 1, 0, 1, 0, 0], [1]
        #     ]
        # ]
        #
        # self.utils.validate_genotype(genotype_1)

        # 1-10-0-1100-0-110010-010001-0
        # 1-10-0-0-0-110000-010001-0
        # 0 - 10 - 100 - 1001 - 0 - 101010 - 000001 - 0 - -- 0 - 10 - 100 - 0 - 0 - 0 - 011000 - 0 - -- 0 - 10 - 0 - 1000 - 0 - 0 - 010100 - 1

        # idl_1 = Individual(genotype_1)
        # idl_1.write_to_file("C:\\Users\\janp\\Desktop", 0)

        # idl_2 = Individual(genotype_2, 2, 4)
        # print(idl_1 == "")
        #
        # # idl = self.population.individuals[0]
        # print(self.population.contains(idl_1))
        # exit()

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
            if self.genes_conn:
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
        self.output_log += "-------------------------------------------------------------------\n"
        self.output_log += "                          STARTUP SETTINGS                         \n"
        self.output_log += "-------------------------------------------------------------------\n"
        self.output_log += "\n"

        self.output_log += "Dataset: "

        if self.dataset.dataset == 0:
            self.output_log += "MNIST\n"
        elif self.dataset.dataset == 1:
            self.output_log += "Fashion MNIST\n"
        elif self.dataset.dataset == 2:
            self.output_log += "SVHN\n"
        elif self.dataset.dataset == 3:
            self.output_log += "CIFAR 10\n"
        elif self.dataset.dataset == 4:
            self.output_log += "CIFAR 100\n"

        self.output_log += "Population: "
        self.output_log += str(self.pop_size) + "\n"
        self.output_log += "Generations: "
        self.output_log += str(self.max_gen) + "\n"
        self.output_log += "Selection: "
        if self.roulette:
            self.output_log += "Roulette\n"
        else:
            self.output_log += "Tournament\n"
            self.output_log += "   Tournament count: "
            self.output_log += str(self.tournament_count) + "\n"
        self.output_log += "Mutation prob.: "
        self.output_log += str(round(self.mutation_probability, 2)) + "\n"
        self.output_log += "Phases: "
        self.output_log += str(self.phase_count) + "\n"
        self.output_log += "Modules: "
        self.output_log += str(self.modules_count) + "\n"
        for i in range(self.modules_count):
            self.output_log += "   " + str(i+1) + ": " + layers.phase_layers[i]["desc"] + "\n"
        self.output_log += "Evolution batch size: "
        self.output_log += str(self.batch_size) + "\n"
        self.output_log += "Evolution epochs: "
        self.output_log += str(self.epochs) + "\n"
        self.output_log += "Validation batch size: "
        self.output_log += str(self.val_batch_size) + "\n"
        self.output_log += "Validation epochs: "
        self.output_log += str(self.val_epochs) + "\n"
        self.output_log += "Validation split: "
        self.output_log += str(self.val_split) + "\n"
        self.output_log += "Verbose: "
        self.output_log += str(self.verbose) + "\n"

        if self.datagen is None:
            self.output_log += "Data augmentation: False\n"
        else:
            self.output_log += "Data augmentation: True\n"
            self.output_log += "   Height shift: "
            self.output_log += str(round(self.datagen.height_shift_range, 2)) + "\n"
            self.output_log += "   Width shift: "
            self.output_log += str(round(self.datagen.width_shift_range, 2)) + "\n"
            self.output_log += "   Rotation range: "
            self.output_log += str(self.datagen.rotation_range) + "\n"
            self.output_log += "   Shear range: "
            self.output_log += str(round(self.datagen.shear_range, 2)) + "\n"
            self.output_log += "   Zoom range: "
            self.output_log += "[" + str(round(self.datagen.zoom_range[0], 2)) + "; " + str(round(self.datagen.zoom_range[1], 2)) + "]\n"
            self.output_log += "   Horizontal flip: "
            self.output_log += str(self.datagen.horizontal_flip) + "\n"
            self.output_log += "   Vertical flip: "
            self.output_log += str(self.datagen.vertical_flip) + "\n"
            self.output_log += "   ZCA Whitening: "
            self.output_log += str(self.datagen.zca_whitening) + "\n"
            self.output_log += "   Featurewise center: "
            self.output_log += str(self.datagen.featurewise_center) + "\n"
            self.output_log += "   Samplewise center: "
            self.output_log += str(self.datagen.samplewise_center) + "\n"
            self.output_log += "   Featurewise std. norm.: "
            self.output_log += str(self.datagen.featurewise_std_normalization) + "\n"
            self.output_log += "   Samplewise std. norm.: "
            self.output_log += str(self.datagen.samplewise_std_normalization) + "\n"

        if not self.result_export.export:
            self.output_log += "Export: None\n"
        else:
            self.output_log += "Export: \n"
            self.output_log += "   Pareto graph: "
            self.output_log += str(self.result_export.pareto_graph) + "\n"
            self.output_log += "   Keras graph: "
            self.output_log += str(self.result_export.keras_graph) + "\n"
            self.output_log += "   Keras model: "
            self.output_log += str(self.result_export.keras_model) + "\n"
            self.output_log += "   Genotype graph/txt: "
            self.output_log += str(self.result_export.genotype_info) + "\n"

        self.output_log += "\n"
        self.output_log += "-------------------------------------------------------------------\n"
        self.output_log += "                             EVOLUTION                             \n"
        self.output_log += "-------------------------------------------------------------------\n"
        self.output_log += "\n"

    def save_result(self):
        print("Saving result...")
        self.utils.save_output_log(self.output_log)
        if self.result_export is not None:
            if self.result_export.export:
                self.utils.save_best_pf(
                    self.population,
                    self.best_pf,
                    self.result_export
                )
        print("Done")
