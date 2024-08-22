from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.xtracker.config import cfg, update_config_from_file


def parameters(yaml_path, weight_path):
    params = TrackerParams()

    # update default config from yaml file
    yaml_file = yaml_path
    update_config_from_file(yaml_file)
    params.cfg = cfg

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    params.num_templates = cfg.DATA.TEMPLATE.NUMBER


    # Network checkpoint path
    params.checkpoint = weight_path
    params.save_all_boxes = False

    return params
