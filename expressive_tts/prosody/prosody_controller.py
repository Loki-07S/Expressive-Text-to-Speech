"""
Prosody Planner Module

Maps emotion and intensity to pitch, energy, and duration scales.
These scales are used to modify the variance predictors in Index-TTS.
"""

class ProsodyController:
    def __init__(self):
        # Default mapping logic
        self.emotion_mapping = {
            "sad": {
                "pitch": lambda intensity: 1 - (0.15 * intensity),
                "energy": lambda intensity: 1 - (0.20 * intensity),
                "duration": lambda intensity: 1 + (0.10 * intensity)
            },
            "happy": {
                "pitch": lambda intensity: 1 + (0.12 * intensity),
                "energy": lambda intensity: 1 + (0.15 * intensity),
                "duration": lambda intensity: 1 - (0.08 * intensity)
            },
            "anger": {
                "pitch": lambda intensity: 1 + (0.20 * intensity),
                "energy": lambda intensity: 1 + (0.30 * intensity),
                "duration": lambda intensity: 1 - (0.15 * intensity)
            },
            "surprise": {
                "pitch": lambda intensity: 1 + (0.25 * intensity),
                "energy": lambda intensity: 1 + (0.10 * intensity),
                "duration": lambda intensity: 1 - (0.05 * intensity)
            },
            "fear": {
                "pitch": lambda intensity: 1 + (0.10 * intensity),
                "energy": lambda intensity: 1 - (0.10 * intensity),
                "duration": lambda intensity: 1 - (0.20 * intensity)
            },
            "disgust": {
                "pitch": lambda intensity: 1 - (0.10 * intensity),
                "energy": lambda intensity: 1 + (0.05 * intensity),
                "duration": lambda intensity: 1 + (0.15 * intensity)
            },
            "neutral": {
                "pitch": lambda intensity: 1.0,
                "energy": lambda intensity: 1.0,
                "duration": lambda intensity: 1.0
            }
        }

    def generate_prosody_scaling(self, emotion, intensity, punctuation=None):
        """
        Calculates prosody scales based on emotion label and intensity.
        
        Args:
            emotion (str): Emotion label from EmotionDetector.
            intensity (float): Intensity score (0 to 1).
            punctuation (list, optional): Optional punctuation tokens for finer control.
            
        Returns:
            dict: pitch_scale, energy_scale, duration_scale
        """
        # Ensure lowercase
        emotion = emotion.lower()
        
        if emotion not in self.emotion_mapping:
            # Fallback to neutral
            emotion = "neutral"
            
        mapping = self.emotion_mapping[emotion]
        
        pitch_scale = mapping["pitch"](intensity)
        energy_scale = mapping["energy"](intensity)
        duration_scale = mapping["duration"](intensity)

        # Placeholder logic for punctuation (e.g. increase duration scale at end of sentence)
        if punctuation and isinstance(punctuation, list):
            if punctuation[-1] in ['.', '!', '?']:
                duration_scale *= 1.05
                
        return {
            "pitch_scale": pitch_scale,
            "energy_scale": energy_scale,
            "duration_scale": duration_scale
        }

if __name__ == "__main__":
    controller = ProsodyController()
    print("Sad (0.82):", controller.generate_prosody_scaling("sad", 0.82))
    print("Happy (0.90):", controller.generate_prosody_scaling("happy", 0.90))
