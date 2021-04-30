class ResultExport:
    def __init__(
            self,
            pareto_graph=False,
            keras_graph=False,
            keras_model=False,
            genotype_info=False
    ):
        self.pareto_graph = pareto_graph
        self.keras_graph = keras_graph
        self.keras_model = keras_model
        self.genotype_info = genotype_info

    def export(self):
        if self.pareto_graph or \
                self.keras_graph or \
                self.keras_model or \
                self.genotype_info:
            return True
        return False

    def export_individual(self):
        if self.keras_graph or \
                self.keras_model or \
                self.genotype_info:
            return True
        return False

    def export_keras(self):
        if self.keras_graph or \
                self.keras_model:
            return True
        return False
