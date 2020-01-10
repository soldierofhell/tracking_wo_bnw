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
        
        # MOT17_Sequence.__getitem__()
        # 1. PIL.Image RGB
        # 2. torchvision.transforms.ToTensor
        
        im = img.cpu().numpy()        
        instances = self.predictor(im)["instances"]
        
        #device = list(self.parameters())[0].device
        #img = img.to(device)

        #detections = self(img)[0]

        return instances.pred_boxes.detach(), instances.scores.detach()

    def predict_boxes(self, images, boxes):
        
        # similar to DefaultPredictor.__call__()

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.predictor.transform_gen.get_transform(original_image).apply_image(original_image)
            
            # BGR -> RGB
            
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
        
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, img):
        pass
