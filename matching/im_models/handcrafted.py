import cv2
import numpy as np
import torch

from matching import BaseMatcher
from matching.utils import to_numpy

from specular_mask import filter_image_feats_with_mask

import time
from types import SimpleNamespace


class HandcraftedBaseMatcher(BaseMatcher):
    """
    This class is the parent for all methods that use a handcrafted detector/descriptor,
    It implements the forward which is the same regardless of the feature extractor of choice.
    Therefore this class should *NOT* be instatiated, as it needs its children to define
    the extractor/detector.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    @staticmethod
    def preprocess(im_tensor: torch.Tensor) -> np.ndarray:
        # convert tensors to np 255-based for openCV
        im_arr = to_numpy(im_tensor).transpose(1, 2, 0)
        im = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        return im

    def _forward(self, img0, img1, mask0=None, mask1=None, logger=None):
        """
        "det_descr" is instantiated by the subclasses.
        """
        # Timing structure
        timings = {
            "extractor_time": 0.0,
            "filter_time": 0.0,
            "matcher_time": 0.0,
            }
        
        # convert tensors to numpy 255-based for OpenCV
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        start_extractor = time.perf_counter()    
        # find the keypoints and descriptors with SIFT
        kp0, des0 = self.det_descr.detectAndCompute(img0, None)
        kp1, des1 = self.det_descr.detectAndCompute(img1, None)
        end_extractor = time.perf_counter()
        timings["extractor_time"] = end_extractor - start_extractor

        # Apply mask filtering
        start_filter = time.perf_counter()
        if mask0 is not None:
            kp0, des0 = filter_image_feats_with_mask(img0, mask0, kp0, des0, logger)
        if mask1 is not None:
            kp1, des1 = filter_image_feats_with_mask(img1, mask1, kp1, des1, logger)
        end_filter = time.perf_counter()
        timings["filter_time"] = end_filter - start_filter

        # Log and return if no keypoints left afetr filtering
        if len(kp0) == 0 or len(kp1) == 0:
            logger.info("No keypoints left after filtering")
            return [], [], [], [], [], [], timings
       
        # match descriptors
        start_matcher = time.perf_counter()
        matches = self.bf.knnMatch(des0, des1, k=self.k_neighbors)


        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good.append(m)

        mkpts0, mkpts1 = [], []
        for good_match in good:
            kpt_0 = kp0[good_match.queryIdx].pt
            kpt_1 = kp1[good_match.trainIdx].pt

            mkpts0.append(kpt_0)
            mkpts1.append(kpt_1)
        
        end_matcher = time.perf_counter()
        timings["matcher_time"] = end_matcher - start_matcher

        mkpts0 = np.array(mkpts0, dtype=np.float32)
        mkpts1 = np.array(mkpts1, dtype=np.float32)

        keypoints_0 = np.array([(x.pt[0], x.pt[1]) for x in kp0])
        keypoints_1 = np.array([(x.pt[0], x.pt[1]) for x in kp1])

        return mkpts0, mkpts1, keypoints_0, keypoints_1, des0, des1, timings

#TODO : See how coudl we modify lowe_thresh and det_descr, bf and k_neighbors. Instead of defining lowe_thresh, get it from the config
class SiftNNMatcher(HandcraftedBaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)

        # Extract parameters from kwargs with defaults from method signature
        params = {
            # Handle parameter name mappings if needed
            "max_num_keypoints": kwargs.get('max_sift_keypoints', max_num_keypoints),
            "lowe_thresh": kwargs.get("lowe_thresh", 0.75),
            "contrast_threshold": kwargs.get("contrast_threshold", 0.04),
            "edge_threshold": kwargs.get("edge_threshold", 10),
            "n_octave_layers": kwargs.get("n_octave_layers", 3),
            "k_neighbors": kwargs.get("k_neighbors", 2)
        }

        # Store config
        self.params = params

        # Create SIFT detector/descriptor
        self.det_descr = cv2.SIFT_create(
            nfeatures=params["max_num_keypoints"],
            contrastThreshold=params["contrast_threshold"],
            edgeThreshold=params["edge_threshold"],
            nOctaveLayers=params["n_octave_layers"]
        )

        # Matcher parameters
        self.threshold = params["lowe_thresh"]
        self.bf = cv2.BFMatcher()
        self.k_neighbors = params["k_neighbors"]
    
    @property
    def extractor_conf(self):
        """Return extractor configuration for reporting"""
        return SimpleNamespace(
            name="SIFT",
            max_num_keypoints=self.params["max_num_keypoints"],
            contrast_threshold=self.params["contrast_threshold"],
            edge_threshold=self.params["edge_threshold"],
            n_octave_layers=self.params["n_octave_layers"]
        )
    
    @property
    def matcher_conf(self):
        """Return matcher configuration for reporting"""
        return SimpleNamespace(
            name="BFMatcher_RatioTest",
            lowe_thresh=self.params["lowe_thresh"],
            k_neighbors=self.params["k_neighbors"],
        )
    
    # Add these properties for API compatibility with LightGlue style
    @property
    def extractor(self):
        return SimpleNamespace(conf=self.extractor_conf)
    
    @property
    def matcher(self):
        return SimpleNamespace(conf=self.matcher_conf)
    
    def __str__(self):
        """Display configuration when the matcher is printed"""
        return f"matcher.extractor.conf\n{self.extractor_conf}\n\nmatcher.matcher.conf\n{self.matcher_conf}"


class OrbNNMatcher(HandcraftedBaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, lowe_thresh=0.75, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.threshold = lowe_thresh
        self.det_descr = cv2.ORB_create(max_num_keypoints)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.k_neighbors = 2
