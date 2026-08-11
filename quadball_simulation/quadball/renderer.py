"""Optional Pygame renderer for interactive match viewing."""

from __future__ import annotations


class PygameRenderer:
    """Render a :class:`QuadballEnvironment` using Pygame.

    Parameters
    ----------
    environment : QuadballEnvironment
        Environment to display.
    pixels_per_metre : int
        Display scale.
    """

    def __init__(self, environment, pixels_per_metre: int = 16) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("Install the 'render' extra to use PygameRenderer.") from exc
        self.pg = pygame
        self.environment = environment
        self.scale = pixels_per_metre
        pygame.init()
        f = environment.config.field
        self.screen = pygame.display.set_mode((int(f.width * self.scale), int(f.height * self.scale)))
        pygame.display.set_caption("Quadball Learning Simulation")
        self.font = pygame.font.SysFont("Arial", 18)
        self.clock = pygame.time.Clock()

    def draw(self) -> bool:
        """Draw one frame and process quit events.

        Returns
        -------
        bool
            False when the user requests window closure; otherwise true.
        """

        pg, env, s = self.pg, self.environment, self.scale
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        self.screen.fill((32, 112, 64))
        pg.draw.line(self.screen, (230, 230, 230), (env.config.field.width * s / 2, 0), (env.config.field.width * s / 2, env.config.field.height * s), 2)
        for team in (0, 1):
            x = env.config.field.goal_x_margin if team == 0 else env.config.field.width - env.config.field.goal_x_margin
            for y in env.config.field.hoop_y:
                pg.draw.circle(self.screen, (240, 220, 80), (int(x * s), int(y * s)), int(env.config.field.hoop_radius * s), 2)
        for player in env.players.values():
            colour = (50, 120, 255) if int(player.team) == 0 else (235, 70, 70)
            if not player.active:
                colour = (110, 110, 110)
            pg.draw.circle(self.screen, colour, tuple((player.position * s).astype(int)), 7)
            label = self.font.render(player.role.name[0], True, (255, 255, 255))
            self.screen.blit(label, tuple((player.position * s + (8, -10)).astype(int)))
        for ball in env.balls:
            colour = (245, 245, 245) if ball.kind.name == "QUAFFLE" else (30, 30, 30)
            pg.draw.circle(self.screen, colour, tuple((ball.position * s).astype(int)), 5)
        if env.flag.active:
            pg.draw.circle(self.screen, (255, 210, 30), tuple((env.flag.position * s).astype(int)), 6)
        score = self.font.render(f"Blue {env.score[0]} - {env.score[1]} Red | {env.time:05.1f}s", True, (255, 255, 255))
        self.screen.blit(score, (10, 8))
        pg.display.flip()
        self.clock.tick(round(1 / env.config.dt))
        return True

    def close(self) -> None:
        """Close the Pygame window and release renderer resources."""
        self.pg.quit()
