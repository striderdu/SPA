from models.dynamic_base_model import DynamicBaseModel
from models.e_spa_tune import ESPATune


class SPATune(DynamicBaseModel):
    def __init__(self, args, dataset_info_dict, device, genotype):
        super(SPATune, self).__init__(args, dataset_info_dict, device)
        self.ent_encoder = ESPATune(genotype, self.args, self.args.embed_size, self.args.hidden_size,
                                              self.args.embed_size,
                                              self.rel_num)
