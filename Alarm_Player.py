import pygame
import threading
import time

class AlarmPlayer:
    """Class to manage non-blocking alarm sound playback using threading."""

    def __init__(self, sound_file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file_path)
        self.is_playing = False

    def play(self):
        """Triggers the alarm sound in a background thread if it is not currently playing."""
        if not self.is_playing:
            threading.Thread(target = self._play_thread, daemon = True).start()

    def _play_thread(self):
        self.is_playing = True
        pygame.mixer.music.play()
        # Keep thread active until the audio finishes playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        self.is_playing = False

    def stop(self):
        """Cleans up audio resources."""
        pygame.mixer.quit()
