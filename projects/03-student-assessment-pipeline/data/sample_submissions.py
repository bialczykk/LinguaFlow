"""Sample student submissions for testing the assessment pipeline.

These are NOT part of the vector store — they are inputs to the pipeline.
Each submission includes the student's writing, the prompt they were given,
and an optional self-reported level hint.
"""


SUBMISSION_B1_TRAVEL = {
    "submission_text": (
        "Traveling is one of the best things you can do in your life. Last year "
        "I traveled to Spain with my friends and it was an amazing experience. "
        "We visited Barcelona and Madrid. In Barcelona we saw the Sagrada Familia "
        "which is a very beautiful church designed by Gaudi. The weather was hot "
        "and sunny every day.\n\n"
        "I think traveling is important because you can learn about different "
        "cultures and meet new people. When you travel you also learn to be more "
        "independent because you need to solve problems by yourself. For example, "
        "in Spain I had to speak Spanish to order food in a restaurant even though "
        "my Spanish is not very good.\n\n"
        "However, traveling can be expensive. Not everyone can afford to go to "
        "other countries. I think governments should help young people to travel "
        "more by offering cheap flights or train tickets.\n\n"
        "In conclusion, traveling is a wonderful way to learn and grow as a person. "
        "I hope I can visit many more countries in the future."
    ),
    "submission_context": "Write an essay about the benefits of traveling.",
    "student_level_hint": "B1",
}

SUBMISSION_A2_HOBBY = {
    "submission_text": (
        "My hobby is play guitar. I started when I was 14 years old. My father "
        "give me a guitar for my birthday. At first it was very difficult because "
        "my fingers was hurting. But I practiced every day and now I can play many "
        "songs.\n\n"
        "I like to play rock music and pop music. My favorite band is Coldplay. "
        "I learn their songs from YouTube videos. Sometimes I play with my friends "
        "and we make a small concert in my house.\n\n"
        "Playing guitar make me happy and relaxed. When I am stressed from school "
        "I play guitar and I feel better. I want to learn more songs and maybe "
        "one day play in a real concert."
    ),
    "submission_context": "Write about your favourite hobby and why you enjoy it.",
    "student_level_hint": "",
}

SUBMISSION_C1_TECHNOLOGY = {
    "submission_text": (
        "The relationship between technological advancement and privacy has become "
        "one of the most pressing issues of the twenty-first century. As digital "
        "technologies permeate every aspect of our lives, the boundaries between "
        "public and private spheres have become increasingly blurred, raising "
        "fundamental questions about the nature of personal autonomy in a "
        "connected world.\n\n"
        "The collection of personal data by corporations and governments has "
        "reached an unprecedented scale. Every online interaction, purchase, and "
        "even physical movement can be tracked, aggregated, and analysed to create "
        "detailed profiles of individuals. Proponents of such surveillance argue "
        "that it is necessary for national security and enables personalised "
        "services that consumers value. However, this argument fails to account "
        "for the power asymmetry it creates — individuals rarely understand the "
        "extent to which their data is being harvested, let alone have meaningful "
        "control over how it is used.\n\n"
        "The implementation of regulations such as the GDPR represents an important "
        "step towards redressing this imbalance, yet enforcement remains inconsistent "
        "and the pace of technological change continually outstrips legislative "
        "responses. What is ultimately needed is a fundamental shift in how we "
        "conceptualise digital rights — treating privacy not as a commodity to be "
        "traded but as an inalienable right that underpins democratic participation."
    ),
    "submission_context": (
        "Discuss the challenges of maintaining personal privacy in the digital age."
    ),
    "student_level_hint": "C1",
}

ALL_SUBMISSIONS = [
    SUBMISSION_B1_TRAVEL,
    SUBMISSION_A2_HOBBY,
    SUBMISSION_C1_TECHNOLOGY,
]
