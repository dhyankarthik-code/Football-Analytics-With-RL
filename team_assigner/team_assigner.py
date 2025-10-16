from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamClassification:
    team: str
    confidence: float
    white_ratio: float
    blue_ratio: float
    mean_color: Tuple[float, float, float]
    bbox: Tuple[int, int, int, int]

class TeamAssigner:
    WHITE_LABEL = "white"
    BLUE_LABEL = "blue"

    def __init__(self, enable_debug: bool = False):
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.player_confidence: Dict[int, float] = {}
        self.player_label: Dict[int, str] = {}
        self.team_label_to_id: Dict[str, int] = {}
        self.team_confidence: Dict[str, float] = {}
        self.debug_enabled = enable_debug
        self.last_debug_samples: List[TeamClassification] = []
        self.kmeans: Optional[KMeans] = None
        self.manual_team_colors: Dict[int, Tuple[int, int, int]] = {}
    
    def set_manual_team_colors(self, team1_color: Tuple[int, int, int], team2_color: Tuple[int, int, int]):
        """Set manual team colors for classification. Team 1 and Team 2."""
        self.manual_team_colors = {1: team1_color, 2: team2_color}
        self.team_colors = self.manual_team_colors.copy()  # Update team_colors for consistency
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None

        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return None

        top_half_image = image[0:int(max(1, image.shape[0]//2)), :]
        if top_half_image.size == 0:
            return None

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color


    def assign_team_color(self,frame, player_detections):
        classifications: List[TeamClassification] = []
        for player_id, player_detection in player_detections.items():
            bbox = player_detection.get("bbox")
            if bbox is None:
                continue
            jersey = self._extract_jersey_region(frame, bbox)
            if jersey.size == 0:
                continue
            classification = self._classify_team_robust(jersey, bbox)
            classifications.append(classification)
            self.player_label[player_id] = classification.team
            self.player_confidence[player_id] = classification.confidence

        if self.debug_enabled:
            self.last_debug_samples = classifications
            self._debug_team_detection(classifications)

        self._resolve_team_mapping(classifications)
        self._populate_team_colors(classifications)
        self._ensure_kmeans_fallback(frame, player_detections)


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        jersey = self._extract_jersey_region(frame, player_bbox)
        classification = self._classify_team_robust(jersey, player_bbox)
        self.player_label[player_id] = classification.team
        self.player_confidence[player_id] = classification.confidence

        team_id = self.team_label_to_id.get(classification.team, 0)
        if team_id == 0 and self.kmeans is not None:
            player_color = self.get_player_color(frame, player_bbox)
            if player_color is None:
                team_id = 0
            else:
                team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id

        return team_id

    def set_debug(self, enabled: bool = True):
        self.debug_enabled = enabled

    def visualize_team_classification(self, frame: np.ndarray, player_detections: Dict[int, Dict]) -> np.ndarray:
        annotated = frame.copy()
        for player_id, player_detection in player_detections.items():
            bbox = player_detection.get("bbox")
            if bbox is None:
                continue
            team_label = self.player_label.get(player_id, "unknown")
            confidence = self.player_confidence.get(player_id, 0.0)
            team_id = self.team_label_to_id.get(team_label)
            color = (128, 128, 128)
            if team_id is not None:
                color = self.team_colors.get(team_id, color)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{team_label} ({confidence:.2f})"
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated

    # --- Internal helpers ---

    def _extract_jersey_region(self, frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = int(max(0, min(w, bbox[0])))
        y1 = int(max(0, min(h, bbox[1])))
        x2 = int(max(0, min(w, bbox[2])))
        y2 = int(max(0, min(h, bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        height = y2 - y1
        jersey_y1 = int(y1 + height * 0.1)
        jersey_y2 = int(y1 + height * 0.5)
        jersey_y1 = max(y1, min(jersey_y1, y2))
        jersey_y2 = max(jersey_y1 + 1, min(jersey_y2, y2))

        jersey = frame[jersey_y1:jersey_y2, x1:x2]
        return jersey

    def _classify_team_robust(self, jersey_region: np.ndarray, bbox) -> TeamClassification:
        bbox_tuple: Tuple[int, int, int, int] = tuple(map(int, bbox[:4]))  # type: ignore[index]
        if jersey_region.size == 0:
            return TeamClassification("unknown", 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), bbox_tuple)

        if self.manual_team_colors:
            # Manual color-based classification
            mean_color_arr = np.mean(jersey_region.reshape(-1, 3), axis=0)
            mean_color = np.array([float(x) for x in mean_color_arr.tolist()[:3]])
            if len(mean_color) != 3:
                mean_color = np.array([0.0, 0.0, 0.0])

            distances = {}
            for team_id, team_color in self.manual_team_colors.items():
                team_color_arr = np.array(team_color)
                dist = np.linalg.norm(mean_color - team_color_arr)
                distances[team_id] = dist

            min_team_id = min(distances.keys(), key=lambda k: distances[k])
            min_dist = distances[min_team_id]
            # Confidence inversely proportional to distance, normalized
            max_possible_dist = np.sqrt(3 * (255**2))  # Max distance in RGB space
            confidence = 1.0 - (min_dist / max_possible_dist)
            team_label = f"team{min_team_id}"
            return TeamClassification(team_label, confidence, 0.0, 0.0, tuple(mean_color), bbox_tuple)

        # Original HSV-based classification
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

        white_pixels = cv2.countNonZero(white_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        total_pixels = float(jersey_region.shape[0] * jersey_region.shape[1])
        if total_pixels == 0:
            return TeamClassification("unknown", 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), bbox_tuple)

        white_ratio = white_pixels / total_pixels
        blue_ratio = blue_pixels / total_pixels
        confidence_threshold = 0.1

        if white_ratio > blue_ratio and white_ratio > confidence_threshold:
            team = self.WHITE_LABEL
            confidence = white_ratio
        elif blue_ratio > white_ratio and blue_ratio > confidence_threshold:
            team = self.BLUE_LABEL
            confidence = blue_ratio
        else:
            team = "unknown"
            confidence = max(white_ratio, blue_ratio)

        mean_color_arr = np.mean(jersey_region.reshape(-1, 3), axis=0)
        mean_color_tuple = tuple(float(x) for x in mean_color_arr.tolist())[:3]
        if len(mean_color_tuple) != 3:
            mean_color_tuple = (0.0, 0.0, 0.0)

        return TeamClassification(team, confidence, white_ratio, blue_ratio, mean_color_tuple, bbox_tuple)

    def _debug_team_detection(self, classifications: List[TeamClassification]):
        print("=== TEAM DETECTION DEBUG ===")
        if not classifications:
            print("No player samples available for classification.")
        for idx, sample in enumerate(classifications):
            print(
                f"Player #{idx} bbox={sample.bbox} team={sample.team} confidence={sample.confidence:.2f}"
                f" | mean_color={sample.mean_color} white_ratio={sample.white_ratio:.2f} blue_ratio={sample.blue_ratio:.2f}"
            )
        print("=== END DEBUG ===")

    def _resolve_team_mapping(self, classifications: List[TeamClassification]):
        if self.manual_team_colors:
            self.team_label_to_id = {"team1": 1, "team2": 2}
            return

        label_counts = Counter([c.team for c in classifications if c.team != "unknown"])

        if len(label_counts) >= 2:
            most_common = [label for label, _ in label_counts.most_common(2)]
            self.team_label_to_id = {most_common[0]: 1, most_common[1]: 2}
        elif len(label_counts) == 1:
            first_label = next(iter(label_counts.keys()))
            fallback_label = self.BLUE_LABEL if first_label == self.WHITE_LABEL else self.WHITE_LABEL
            self.team_label_to_id = {first_label: 1, fallback_label: 2}
        else:
            self.team_label_to_id = {self.WHITE_LABEL: 1, self.BLUE_LABEL: 2}

    def _populate_team_colors(self, classifications: List[TeamClassification]):
        if self.manual_team_colors:
            # Colors already set manually
            for label, team_id in self.team_label_to_id.items():
                label_samples = [c for c in classifications if c.team == label]
                if label_samples:
                    self.team_confidence[label] = float(np.mean([sample.confidence for sample in label_samples]))
                else:
                    self.team_confidence[label] = 0.0
            return

        for label, team_id in self.team_label_to_id.items():
            label_samples = [c for c in classifications if c.team == label]
            if label_samples:
                mean_color = np.mean([sample.mean_color for sample in label_samples], axis=0)
                mean_color_tuple = tuple(int(np.clip(float(x), 0, 255)) for x in mean_color.tolist())[:3]
                if len(mean_color_tuple) == 3:
                    self.team_colors[team_id] = mean_color_tuple
                self.team_confidence[label] = float(np.mean([sample.confidence for sample in label_samples]))
            else:
                default_color = (255, 255, 255) if label == self.WHITE_LABEL else (255, 0, 0)
                self.team_colors[team_id] = default_color
                self.team_confidence[label] = 0.0

        for fallback_label, team_id in [(self.WHITE_LABEL, 1), (self.BLUE_LABEL, 2)]:
            if team_id not in self.team_colors:
                self.team_colors[team_id] = (255, 255, 255) if fallback_label == self.WHITE_LABEL else (255, 0, 0)
                self.team_confidence[fallback_label] = 0.0

    def _ensure_kmeans_fallback(self, frame: np.ndarray, player_detections: Dict[int, Dict]):
        if self.team_label_to_id and len(self.team_label_to_id) >= 2:
            return

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection.get("bbox")
            if bbox is None:
                continue
            player_color = self.get_player_color(frame,bbox)
            if player_color is not None:
                player_colors.append(player_color)

        if len(player_colors) < 2:
            defaults = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.float32)
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
            kmeans.fit(defaults)
            self.kmeans = kmeans
            self.team_colors[1] = (
                int(defaults[0][0]),
                int(defaults[0][1]),
                int(defaults[0][2]),
            )
            self.team_colors[2] = (
                int(defaults[1][0]),
                int(defaults[1][1]),
                int(defaults[1][2]),
            )
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        center0 = kmeans.cluster_centers_[0]
        center1 = kmeans.cluster_centers_[1]
        self.team_colors[1] = (int(center0[0]), int(center0[1]), int(center0[2]))
        self.team_colors[2] = (int(center1[0]), int(center1[1]), int(center1[2]))
