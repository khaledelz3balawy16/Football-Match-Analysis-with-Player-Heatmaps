import numpy as np
import supervision as sv
class resolve_goalkeepers:
    def resolve_goalkeepers_team_id(
        players: sv.Detections,
        goalkeepers: sv.Detections
    ) -> np.ndarray:
        # Get coordinates of goalkeepers and players
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        # Calculate centroids for both teams
        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0) if (players.class_id == 0).any() else np.array([0, 0])
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0) if (players.class_id == 1).any() else np.array([0, 0])

        # Assign team IDs based on proximity to centroids
        goalkeepers_team_id = [
            0 if np.linalg.norm(gk_xy - team_0_centroid) < np.linalg.norm(gk_xy - team_1_centroid) else 1
            for gk_xy in goalkeepers_xy
        ]

        return np.array(goalkeepers_team_id)
