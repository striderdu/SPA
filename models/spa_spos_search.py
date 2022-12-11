from models.dynamic_base_model import DynamicBaseModel
from models.e_spa_spos_search import ESPASPOSSearch


class SPASPOSSearch(DynamicBaseModel):
    def __init__(self, args, dataset_info_dict, device):
        super(SPASPOSSearch, self).__init__(args, dataset_info_dict, device)
        self.ent_encoder = ESPASPOSSearch(self.args, self.args.embed_size, self.args.hidden_size,
                                           self.args.embed_size,
                                           self.rel_num)
