import numpy as np
import pygame


class Vizualizer:
    def __init__(self):
        self.bars = []
        frequencies = np.arange(100, 8000, 100)
        total_width = 800  # Total width of the area where you're displaying the bars
        num_bars = len(frequencies)  # Number of bars
        width = total_width / num_bars  # Width of each bar
        x = 0  # Initial x position
        for c in frequencies:
            # self.bars.append(AudioBar(x, 300, c, (255, 0, 0), max_height=400, width=width))
            x += width

    def render(self, screen, audio):
        t = pygame.time.get_ticks()
        getTicksLastFrame = t
        pygame.mixer.music.load(self.filename)
        pygame.mixer.music.play(0)

        # Run until the user asks to quit
        running = True
        while running:
            t = pygame.time.get_ticks()
            deltaTime = (t - getTicksLastFrame) / 1000.0
            getTicksLastFrame = t

            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Fill the background with white
            screen.fill((255, 255, 255))

            for b in self.bars:
                b.update(
                    deltaTime,
                    audio.get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq),
                )
                b.render(screen)

            # Flip the display
            pygame.display.flip()

        # Done! Time to quit.
        pygame.quit()
