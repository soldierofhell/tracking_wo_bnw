import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


class CRCNN_FPN():

    def __init__(self, num_classes=1):
        cfg = get_cfg()
        cfg.merge_from_file("/content/detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
        cfg.MODEL.WEIGHTS = "/content/tracking_wo_bnw/model_final.pth"
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1        
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

        self.model = build_model(cfg)
        self.model.eval()
        self.model.cuda()
        
        self.proposal_generator = self.model.proposal_generator
        
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def detect(self, img, img_size):
        
        self.model.proposal_generator = self.proposal_generator
        
        device = list(self.model.parameters())[0].device
        img = img[0].to(device)
        height = 1080 # img_size[0].numpy() #to(device)
        width = 1920 # img_size[1].numpy() #.to(device)
        
        print('img size: ', img_size)
        
        inputs = {"image": img, "height": height, "width": width}
        with torch.no_grad():
            instances = self.model([inputs])[0]["instances"]
            
            pred_boxes = instances.pred_boxes.tensor.detach()
            pred_scores = instances.scores.detach()

        return pred_boxes, pred_scores

    def predict_boxes(self, images, boxes, img_size):
        
        self.model.proposal_generator = None
        
        device = list(self.model.parameters())[0].device
        img = images[0].to(device)
        height = 1080 # img_size[0].numpy() #to(device)
        width = 1920 # img_size[1].numpy() #.to(device)
        
        print('img size: ', img.size())
        
        inputs = {"image": img, "height": height, "width": width, "proposals": boxes}
        with torch.no_grad():
            instances = self.model([inputs])[0]["instances"]
            
            pred_boxes = instances.pred_boxes.tensor.detach()
            pred_scores = instances.scores.detach()
        
        return pred_boxes, pred_scores

    def load_image(self, img):
        pass
