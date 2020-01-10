import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


class CRCNN_FPN():

    def __init__(self, num_classes=1):
        cfg = get_cfg()
        cfg.merge_from_file("detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl"
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1        
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.model = build_model(cfg)
        self.model.eval()
        
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def detect(self, img):
        
        device = list(self.model.parameters())[0].device
        img = img.to(device)
        
        inputs = {"image": img, "height": img.size(0), "width": img.size(1)}
        with torch.no_grad():
            instances = self.predictor(inputs)["instances"]

        return instances.pred_boxes.detach(), instances.scores.detach()

    def predict_boxes(self, images, boxes):
        
        device = list(self.model.parameters())[0].device
        img = img.to(device)
        
        inputs = {"image": img, "height": img.size(0), "width": img.size(1)}
        with torch.no_grad():
            instances = self.predictor(inputs)["instances"]
        
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, img):
        pass
