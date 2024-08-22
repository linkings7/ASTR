class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/Tracker/ASTR_pub'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/Tracker/ASTR_pub/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/Tracker/ASTR_pub/pretrained_networks'
        self.got10k_val_dir = '/Tracker/ASTR_pub/data/got10k/val'
        self.lasot_lmdb_dir = '/Tracker/ASTR_pub/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/Tracker/ASTR_pub/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/Tracker/ASTR_pub/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/Tracker/ASTR_pub/data/coco_lmdb'
        self.coco_dir = '/Tracker/ASTR_pub/data/coco'
        self.lasot_dir = '/Tracker/ASTR_pub/data/lasot'
        self.got10k_dir = '/Tracker/ASTR_pub/data/got10k/train'
        self.trackingnet_dir = '/Tracker/ASTR_pub/data/trackingnet'
        self.depthtrack_dir = '/Tracker/ASTR_pub/data/depthtrack/train'
        self.lasher_dir = '/Tracker/ASTR_pub/data/lasher/trainingset'
        self.visevent_dir = '/Tracker/ASTR_pub/data/visevent/train'
