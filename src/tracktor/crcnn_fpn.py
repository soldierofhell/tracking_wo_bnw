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

        self.predictor = DefaultPredictor(cfg)

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def predict_boxes(self, images, boxes):
        device = list(self.parameters())[0].device
        images = images.to(device)
        boxes = boxes.to(device)
        
        print('boxes: ', boxes.size(), boxes)

        targets = None
        original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        # proposals, proposal_losses = self.rpn(images, features, targets)
        from torchvision.models.detection.transform import resize_boxes
        boxes = resize_boxes(
            boxes, original_image_sizes[0], images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, images.image_sizes[0], original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, img):
        pass
