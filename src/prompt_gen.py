class PromptOptimizer:
    @staticmethod
    def format_suno_prompt(features, tags, custom_additions=None):
        components = []

        # 1. Genre (highest confidence)
        genre_tags = [t for t in tags if t["category"] == "genres"]
        if genre_tags:
            components.append(genre_tags[0]["term"])

        # 2. Tempo
        tempo = features.get("tempo_bpm")
        if tempo:
            components.append(f"{tempo:.0f}bpm")

        # 3. Mood
        mood_tags = [t for t in tags if t["category"] == "moods"]
        if mood_tags:
            components.append(mood_tags[0]["term"])

        # 4. Instruments & Techniques (up to 2 each)
        inst_tags = [t for t in tags if t["category"] == "instruments"][:2]
        tech_tags = [t for t in tags if t["category"] == "techniques"][:2]
        for t in inst_tags + tech_tags:
            components.append(t["term"])

        # 5. Custom additions
        if custom_additions:
            components.extend(custom_additions)

        # Deduplicate & limit to 6-7 components (optimal for Suno/Udio)
        seen = set()
        unique_components = []
        for c in components:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_components.append(c)
                if len(unique_components) >= 7:
                    break

        return ", ".join(unique_components)
