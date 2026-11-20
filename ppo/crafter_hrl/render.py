"""Visualización de episodios: grabación de video y ventana en vivo.

`Renderer` se engancha como callback `on_env_step` de `PlannerEnv` (o se llama
a mano en cualquier loop con `CrafterEnv`) y reparte cada frame entre sus
salidas: un `VideoRecorder` (.mp4/.gif) y/o una `LiveWindow` (pygame).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .env import CrafterEnv


class FrameSink(Protocol):
    def add(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


class VideoRecorder:
    """Acumula frames en memoria y los guarda al cerrar (.mp4, o .gif si no hay ffmpeg)."""

    def __init__(self, path: str | Path, fps: int = 15):
        self.path, self.fps = Path(path), fps
        self.frames: list[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        if not self.frames:
            return
        import imageio

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            imageio.mimsave(self.path, self.frames, **self._save_kwargs(self.path))
        except ValueError:  # sin backend para mp4 (falta imageio[ffmpeg])
            self.path = self.path.with_suffix(".gif")
            print("Sin soporte mp4 (pip install 'imageio[ffmpeg]'): guardo GIF.")
            imageio.mimsave(self.path, self.frames, **self._save_kwargs(self.path))
        print(f"\nVideo guardado en {self.path} ({len(self.frames)} frames)")

    def _save_kwargs(self, path: Path) -> dict[str, float]:
        if path.suffix == ".mp4":
            return {"fps": self.fps}
        return {"duration": 1000 / self.fps}


class LiveWindow:
    """Ventana pygame que muestra cada frame a `fps` cuadros por segundo."""

    def __init__(self, size: int, fps: int = 15):
        import pygame

        pygame.init()
        self.pygame, self.fps = pygame, fps
        self.screen = pygame.display.set_mode((size, size))
        self.clock = pygame.time.Clock()

    def add(self, frame: np.ndarray) -> None:
        surf = self.pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self.screen.blit(surf, (0, 0))
        self.pygame.display.flip()
        self.pygame.event.pump()
        self.clock.tick(self.fps)

    def close(self) -> None:
        self.pygame.quit()


class Renderer:
    """Callback `(env, action, info)` que renderiza el entorno y lo manda a cada salida."""

    def __init__(self, sinks: list[FrameSink], size: int = 1024):
        self.sinks, self.size = sinks, size

    @classmethod
    def from_options(cls, video: str | None = None, window: bool = False,
                     fps: int = 15, size: int = 1024) -> Renderer:
        sinks: list[FrameSink] = []
        if video:
            sinks.append(VideoRecorder(video, fps))
        if window:
            sinks.append(LiveWindow(size, fps))
        return cls(sinks, size)

    def __call__(self, env: CrafterEnv, action: int, info: dict) -> None:
        if not self.sinks:
            return
        frame = env.render(self.size)
        for sink in self.sinks:
            sink.add(frame)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
