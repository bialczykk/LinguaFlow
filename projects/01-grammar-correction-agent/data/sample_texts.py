"""Sample student writing at various CEFR levels for testing and demonstration.

Each sample is a dict with:
- label: short description including expected CEFR level
- text: the student's writing (with deliberate errors matching the level)

These are used in the interactive CLI to let users quickly try the agent
without typing their own text.
"""

SAMPLE_TEXTS = [
    {
        "label": "A2 (Elementary) — Daily routine description",
        "text": (
            "Every day I wake up at 7 clock and I go to the school. "
            "I have breakfast with my family, we eat breads and drink milk. "
            "After school I play with my friend in the park. "
            "Yesterday I go to the cinema and watch a very good film. "
            "I am like learning English because it is important for my future."
        ),
    },
    {
        "label": "B1 (Intermediate) — Opinion essay on technology",
        "text": (
            "In my opinion, technology have changed our lifes in many ways. "
            "Most of people today cannot imagine their daily routine without "
            "smartphones. I think that social media is both good and bad for "
            "the society. On one hand, it help us to stay connected with friends "
            "who lives far away. On the other hand, many peoples spend too much "
            "time scrolling and this affect their mental health. In conclusion, "
            "we should to use technology wisely and not let it control us."
        ),
    },
    {
        "label": "B2 (Upper-intermediate) — Formal email to a professor",
        "text": (
            "Dear Professor Smith,\n\n"
            "I am writing to you in regards of the assignment that was due last "
            "Friday. Unfortunately, I was not able to submit it on time due to "
            "an unexpected family emergency which happened me last week. I have "
            "already completed most of the work and I would be very grateful if "
            "you could extend the deadline for few more days. I assure you that "
            "the quality of my work will not be effected by this delay. I look "
            "forward to hear from you.\n\n"
            "Best regards,\nMaria"
        ),
    },
    {
        "label": "C1 (Advanced) — Academic paragraph on climate change",
        "text": (
            "The ramifications of climate change extend far beyond the environmental "
            "sphere, permeating into economic stability and social cohesion. While "
            "some argue that the transition to renewable energy sources will "
            "inevitably lead to job losses in traditional industries, this perspective "
            "fails to account the burgeoning green economy which has already began "
            "generating millions of positions worldwide. Furthermore, the cost of "
            "inaction — measured in terms of natural disasters, healthcare expenditures, "
            "and agricultural disruptions — far outweights the investment required for "
            "a comprehensive energy transition. It is therefore imperative that policy "
            "makers adopt a more holistic approach which takes into consideration both "
            "the immediate economic concerns and long-term sustainability objectives."
        ),
    },
]
