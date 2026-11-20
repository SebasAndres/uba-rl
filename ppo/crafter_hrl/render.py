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
    """Destino de frames (video, ventana, ...)."""

    def add(self, frame: np.ndarray) -> None:
        """Recibe un frame.

        Parameters
        ----------
        frame : np.ndarray
            Imagen RGB (H, W, 3) uint8.
        """
        ...

    def close(self) -> None:
        """Libera recursos / escribe a disco."""
        ...


class VideoRecorder:
    """Acumula frames en memoria y los guarda al cerrar (.mp4, o .gif si no hay ffmpeg).

    Parameters
    ----------
    path : str or Path
        Archivo de salida.
    fps : int, optional
        Cuadros por segundo.
    """

    def __init__(self, path: str | Path, fps: int = 15) -> None:
        self.path, self.fps = Path(path), fps
        self.frames: list[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        """Agrega un frame al video.

        Parameters
        ----------
        frame : np.ndarray
            Imagen RGB (H, W, 3) uint8.
        """
        self.frames.append(frame)

    def close(self) -> None:
        """Escribe el video; si no hay soporte para mp4, cae a GIF."""
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
        """Argumentos de velocidad para `imageio.mimsave` según el formato.

        Parameters
        ----------
        path : Path
            Archivo de salida.

        Returns
        -------
        dict
            {"fps": ...} para mp4 o {"duration": ms por frame} para GIF.
        """
        if path.suffix == ".mp4":
            return {"fps": self.fps}
        return {"duration": 1000 / self.fps}


class LiveWindow:
    """Ventana pygame que muestra cada frame a `fps` cuadros por segundo.

    Parameters
    ----------
    size : int
        Lado de la ventana en píxeles.
    fps : int, optional
        Cuadros por segundo.
    """

    def __init__(self, size: int, fps: int = 15) -> None:
        import pygame

        pygame.init()
        self.pygame, self.fps = pygame, fps
        self.screen = pygame.display.set_mode((size, size))
        self.clock = pygame.time.Clock()

    def add(self, frame: np.ndarray) -> None:
        """Dibuja un frame y espera para respetar los fps.

        Parameters
        ----------
        frame : np.ndarray
            Imagen RGB (H, W, 3) uint8.
        """
        surface = self.pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # pygame usa (W, H)
        self.screen.blit(surface, (0, 0))
        self.pygame.display.flip()
        self.pygame.event.pump()
        self.clock.tick(self.fps)

    def close(self) -> None:
        """Cierra la ventana."""
        self.pygame.quit()


class Renderer:
    """Callback `(env, action, info)` que renderiza el entorno y lo manda a cada salida.

    Parameters
    ----------
    sinks : list of FrameSink
        Salidas (video, ventana). Si está vacía, no se renderiza nada.
    size : int, optional
        Lado del frame renderizado en píxeles.
    """

    def __init__(self, sinks: list[FrameSink], size: int = 1024) -> None:
        self.sinks, self.size = sinks, size

    @classmethod
    def from_options(cls, video: str | None = None, window: bool = False,
                     fps: int = 15, size: int = 1024) -> Renderer:
        """Arma el renderer a partir de los flags de línea de comandos.

        Parameters
        ----------
        video : str or None, optional
            Archivo de video a grabar.
        window : bool, optional
            Mostrar una ventana en vivo.
        fps : int, optional
            Cuadros por segundo.
        size : int, optional
            Lado del frame en píxeles.

        Returns
        -------
        Renderer
            Renderer con las salidas pedidas.
        """
        sinks: list[FrameSink] = []
        if video:
            sinks.append(VideoRecorder(video, fps))
        if window:
            sinks.append(LiveWindow(size, fps))
        return cls(sinks, size)

    def __call__(self, env: CrafterEnv, action: int, info: dict) -> None:
        """Renderiza el estado actual y lo manda a todas las salidas.

        Parameters
        ----------
        env : CrafterEnv
            Entorno a renderizar.
        action : int
            Acción recién ejecutada (no se usa).
        info : dict
            `info` de Crafter (no se usa).
        """
        if not self.sinks:
            return
        frame = env.render(self.size)
        for sink in self.sinks:
            sink.add(frame)

    def close(self) -> None:
        """Cierra todas las salidas (guarda el video, cierra la ventana)."""
        for sink in self.sinks:
            sink.close()

    def __enter__(self) -> Renderer:
        """Permite usarlo con `with`."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Cierra las salidas al salir del `with`."""
        self.close()
