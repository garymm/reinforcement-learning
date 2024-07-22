from acme.wrappers import video


class VideoWrapper(video.VideoWrapper):
    """Like the base class but supports frame unstacking."""

    def _render_frame(self, observation):
        """Renders a frame from the given environment observation."""
        if observation.ndim == 3:
            return observation
        elif observation.ndim == 4:
            return observation[0]
        else:
            raise ValueError(
                f"Expected observation to have rank 3 or 4. Got rank {observation.ndim}"
            )
