"""Sample scored essays for each CEFR level.

These are stored in the Chroma vector store and retrieved during assessment
as reference examples. Seeing how a scored essay looks at the same level
helps the LLM calibrate its scoring.

12 documents: 2 per CEFR level (A1-C2).
"""

from langchain_core.documents import Document

# -- A1 Sample Essays --

A1_ESSAY_1 = Document(
    page_content=(
        "CEFR Level A1 — Sample Essay 1 (Score: 2/5)\n\n"
        "Prompt: Write about your family.\n\n"
        "Student Essay:\n"
        "My family. I have mother and father. My mother name is Anna. "
        "She is teacher. My father work in office. I have one sister. "
        "She is 10. We live in house. House is big. I like my family.\n\n"
        "Assessor Notes:\n"
        "Very limited vocabulary and highly formulaic. Missing articles throughout "
        "(e.g., 'My mother name' instead of 'My mother's name'). Simple sentences "
        "with frequent grammar omissions. Meaning is barely communicated. "
        "Addresses the prompt but with minimal development."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A1", "score": 2},
)

A1_ESSAY_2 = Document(
    page_content=(
        "CEFR Level A1 — Sample Essay 2 (Score: 4/5)\n\n"
        "Prompt: Write about your favourite food.\n\n"
        "Student Essay:\n"
        "My favourite food is pizza. I like pizza very much. Pizza is from Italy. "
        "It has cheese and tomato. Sometimes it has mushrooms. I eat pizza on "
        "Saturday with my family. My mother makes pizza at home. It is very good. "
        "I don't like fish. Fish is not good for me.\n\n"
        "Assessor Notes:\n"
        "Good control of basic A1 structures. Simple sentences are mostly correct. "
        "Some minor errors ('It has cheese' is slightly unusual but acceptable at this level). "
        "Limited but appropriate vocabulary for the task. Ideas are logically sequenced. "
        "Addresses the prompt fully with adequate basic detail."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A1", "score": 4},
)

# -- A2 Sample Essays --

A2_ESSAY_1 = Document(
    page_content=(
        "CEFR Level A2 — Sample Essay 1 (Score: 2/5)\n\n"
        "Prompt: Describe your daily routine.\n\n"
        "Student Essay:\n"
        "Every day I wake up at 7 o'clock. I eat breakfast and go to school. "
        "At school I study English, maths and science. After school I go home "
        "and do homework. Then I watch TV. I go to sleep at 10. "
        "On weekend I play football with friends. I like weekend more than school days.\n\n"
        "Assessor Notes:\n"
        "Very basic vocabulary with little variety. Sentences are all simple and "
        "follow the same subject-verb-object pattern. No linking devices beyond "
        "simple sequence (Then). Grammar is mostly accurate for simple structures "
        "but no attempt at more complex forms. The prompt is addressed but the "
        "response lacks detail and development expected at A2."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A2", "score": 2},
)

A2_ESSAY_2 = Document(
    page_content=(
        "CEFR Level A2 — Sample Essay 2 (Score: 4/5)\n\n"
        "Prompt: Write about a place you like to visit.\n\n"
        "Student Essay:\n"
        "My favourite place to visit is the park near my house. It is a big park "
        "with many trees and flowers. In summer the park is very beautiful because "
        "all the flowers are open. I go there every weekend with my dog, Max. "
        "We walk for one hour and sometimes we sit near the lake. "
        "I like the park because it is quiet and I can relax there. "
        "There is also a cafe in the park where I drink coffee. "
        "When I am in the park, I feel happy and free.\n\n"
        "Assessor Notes:\n"
        "Good range of A2 vocabulary with some variety. Uses 'because' to explain "
        "reasons, which shows emerging cohesion. Mostly accurate grammar with only "
        "minor errors ('all the flowers are open' is slightly unusual). "
        "Good development of ideas with specific details (the dog, the lake, the cafe). "
        "Well-organized for the level."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A2", "score": 4},
)

# -- B1 Sample Essays --

B1_ESSAY_1 = Document(
    page_content=(
        "CEFR Level B1 — Sample Essay 1 (Score: 3/5)\n\n"
        "Prompt: Do you think social media has more benefits or disadvantages?\n\n"
        "Student Essay:\n"
        "In my opinion, social media has both benefits and disadvantages, but I think "
        "the disadvantages are more important. Firstly, social media can be addictive. "
        "Many young people spend too much time on their phones instead of doing "
        "homework or sport. Also, there is a lot of false information on social media "
        "and it is difficult to know what is true. However, social media is also useful "
        "because you can communicate with friends and family who live far away. "
        "You can also find information and news quickly. In conclusion, I think "
        "we should use social media carefully and not too much.\n\n"
        "Assessor Notes:\n"
        "Reasonable B1 essay with a clear position and some supporting points. "
        "Uses basic discourse markers (Firstly, Also, However, In conclusion). "
        "Vocabulary is adequate but lacks variety — 'important' and 'useful' are "
        "overused. Grammar is mostly correct with occasional simplicity. "
        "Arguments could be more developed with specific examples."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B1", "score": 3},
)

B1_ESSAY_2 = Document(
    page_content=(
        "CEFR Level B1 — Sample Essay 2 (Score: 5/5)\n\n"
        "Prompt: Write about a challenge you have overcome.\n\n"
        "Student Essay:\n"
        "Three years ago, I decided to learn how to swim even though I was terrified "
        "of water. When I was a child, I nearly drowned at the beach, and since then "
        "I had always avoided swimming pools and the sea.\n\n"
        "However, I realized that my fear was stopping me from enjoying many things. "
        "For example, my friends often went to the lake in summer and I always had "
        "to make excuses. So I signed up for adult swimming classes at the local pool.\n\n"
        "At first, it was extremely difficult. I felt anxious every time I got into "
        "the water, and I wanted to give up after the first lesson. But my instructor "
        "was very patient and encouraging. Gradually, my confidence grew.\n\n"
        "After three months, I could swim 20 lengths of the pool. Last summer, "
        "I finally went to the lake with my friends. It was one of the best days "
        "of my life. This experience taught me that facing your fears is always "
        "worth it in the end.\n\n"
        "Assessor Notes:\n"
        "Excellent B1 essay. Well-organized with clear narrative progression. "
        "Good range of vocabulary (terrified, anxious, gradually, encouraging). "
        "Uses complex sentences and a variety of tenses accurately. "
        "Linking devices are used skillfully (However, For example, But, Gradually). "
        "Fully addresses the prompt with strong development and a clear conclusion."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B1", "score": 5},
)

# -- B2 Sample Essays --

B2_ESSAY_1 = Document(
    page_content=(
        "CEFR Level B2 — Sample Essay 1 (Score: 3/5)\n\n"
        "Prompt: 'Universities should focus on practical skills rather than "
        "theoretical knowledge.' Discuss.\n\n"
        "Student Essay:\n"
        "The debate about whether universities should prioritise practical skills or "
        "theoretical knowledge has become increasingly relevant in today's job market. "
        "While some argue that practical training better prepares students for employment, "
        "others believe that theoretical grounding is essential for deeper understanding.\n\n"
        "On one hand, employers often complain that graduates lack the practical skills "
        "needed in the workplace. Students who spend years studying theory may struggle "
        "to apply their knowledge in real situations. For instance, a business student "
        "who has never created a real budget may find their first job challenging.\n\n"
        "On the other hand, theoretical knowledge provides the foundation that allows "
        "professionals to adapt to new situations. Without understanding the principles "
        "behind their field, workers may be unable to solve novel problems.\n\n"
        "In conclusion, universities should aim to balance both approaches rather than "
        "choosing one over the other.\n\n"
        "Assessor Notes:\n"
        "Competent B2 essay with clear structure. Uses appropriate academic vocabulary "
        "and discourse markers. However, arguments are somewhat generic and lack "
        "specific evidence. Grammar is mostly accurate but sentence structures "
        "are not particularly varied. A good but unremarkable B2 response."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B2", "score": 3},
)

B2_ESSAY_2 = Document(
    page_content=(
        "CEFR Level B2 — Sample Essay 2 (Score: 5/5)\n\n"
        "Prompt: Should cities invest more in public transport or in road infrastructure?\n\n"
        "Student Essay:\n"
        "As urban populations continue to grow, city planners face a critical choice: "
        "should limited budgets be directed towards expanding public transport networks "
        "or improving road infrastructure? I would argue that investment in public "
        "transport is not only more effective but also more equitable.\n\n"
        "The environmental case for public transport is compelling. A single bus can "
        "carry 50 passengers who would otherwise drive individual cars, significantly "
        "reducing carbon emissions and air pollution. Cities like Amsterdam and Zurich "
        "demonstrate that efficient, affordable public transport can dramatically "
        "reduce car dependency.\n\n"
        "Furthermore, road expansion tends to generate its own demand — a phenomenon "
        "known as induced demand — meaning that new roads quickly fill with traffic, "
        "providing only temporary relief. Public transport, by contrast, can move "
        "large numbers of people efficiently without this problem.\n\n"
        "Critics argue that not everyone can rely on public transport, particularly "
        "in suburban or rural areas. This is a valid concern, and targeted road "
        "improvements in such areas remain necessary. However, for the majority of "
        "urban residents, the case for investing in buses, trams, and trains "
        "is overwhelming.\n\n"
        "Assessor Notes:\n"
        "Outstanding B2 essay. Sophisticated vocabulary (equitable, compelling, "
        "induced demand, dependency). Well-structured argument with specific examples "
        "and a nuanced acknowledgement of counter-arguments. Grammar is consistently "
        "accurate with a good variety of complex structures. Demonstrates strong "
        "critical thinking."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B2", "score": 5},
)

# -- C1 Sample Essays --

C1_ESSAY_1 = Document(
    page_content=(
        "CEFR Level C1 — Sample Essay 1 (Score: 3/5)\n\n"
        "Prompt: Evaluate the claim that economic growth is incompatible with "
        "environmental sustainability.\n\n"
        "Student Essay:\n"
        "The tension between economic growth and environmental sustainability has "
        "been a subject of intense debate among economists and environmentalists alike. "
        "The traditional view holds that growth inevitably depletes natural resources "
        "and generates pollution, making the two goals fundamentally incompatible. "
        "However, proponents of sustainable development argue that green technologies "
        "and circular economies can decouple growth from environmental damage.\n\n"
        "There is considerable evidence supporting the pessimistic view. Historical "
        "data shows that rising GDP has consistently been associated with increased "
        "carbon emissions and biodiversity loss. Despite technological advances, "
        "global resource consumption continues to exceed planetary boundaries.\n\n"
        "Nevertheless, the emergence of renewable energy and more efficient production "
        "methods suggests that the relationship between growth and environmental impact "
        "is not fixed. Some Nordic countries have achieved economic growth while "
        "reducing their carbon footprints, suggesting that the decoupling thesis "
        "may hold in certain contexts.\n\n"
        "In conclusion, while the evidence suggests that unlimited growth cannot be "
        "truly sustainable, a carefully managed transition to green economies may "
        "offer a more nuanced resolution to this apparent dilemma.\n\n"
        "Assessor Notes:\n"
        "Solid C1 essay with good academic vocabulary and a balanced argument. "
        "However, ideas could be more fully developed and the analysis lacks the "
        "depth expected at the top of C1. Some discourse management is effective "
        "but transitions between ideas could be smoother."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C1", "score": 3},
)

C1_ESSAY_2 = Document(
    page_content=(
        "CEFR Level C1 — Sample Essay 2 (Score: 5/5)\n\n"
        "Prompt: To what extent does language shape our perception of reality?\n\n"
        "Student Essay:\n"
        "The Sapir-Whorf hypothesis — the proposition that the language we speak "
        "fundamentally shapes the way we perceive and conceptualise reality — has "
        "fascinated linguists and philosophers for nearly a century. While the strong "
        "version of the theory, which holds that language determines thought entirely, "
        "has been largely discredited, the weaker claim that language influences "
        "cognition in subtle yet meaningful ways commands increasing empirical support.\n\n"
        "Evidence from colour perception studies is particularly instructive. "
        "Languages vary considerably in the number of colour terms they possess, "
        "and research has demonstrated that speakers of languages with richer colour "
        "vocabularies can discriminate between shades more rapidly. This suggests "
        "that linguistic categories do not merely label pre-existing perceptual "
        "distinctions but actively sharpen them.\n\n"
        "Similarly, languages differ in how they encode spatial relationships, "
        "time, and agency. The Hopi language, famously, was claimed to have no "
        "tense distinctions — though this claim has since been challenged — while "
        "languages such as Guugu Yimithirr use absolute rather than relative spatial "
        "orientation, with speakers demonstrating remarkable navigational abilities "
        "as a consequence.\n\n"
        "What emerges from this evidence is not linguistic determinism but rather "
        "a more nuanced picture: language provides cognitive scaffolding that both "
        "enables and constrains thought without wholly determining it. To speak a "
        "language is to inhabit a particular conceptual architecture — one that can "
        "be transcended with effort but whose influence is pervasive and often "
        "invisible to those within it.\n\n"
        "Assessor Notes:\n"
        "Exceptional C1 essay. Sophisticated academic vocabulary deployed with "
        "precision (empirical support, commands, linguistic determinism, cognitive "
        "scaffolding). Complex grammatical structures used accurately and elegantly. "
        "Argument is well-structured, nuanced, and draws on specific evidence. "
        "The final paragraph is particularly impressive in its synthesis."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C1", "score": 5},
)

# -- C2 Sample Essays --

C2_ESSAY_1 = Document(
    page_content=(
        "CEFR Level C2 — Sample Essay 1 (Score: 3/5)\n\n"
        "Prompt: 'Art serves no practical purpose and should not receive public funding.' "
        "Critically assess this view.\n\n"
        "Student Essay:\n"
        "The claim that art is merely decorative and therefore undeserving of public "
        "subsidy rests on a reductive conception of value — one that conflates worth "
        "with economic utility. While it is true that a painting or a symphony cannot "
        "build a bridge or cure a disease, this narrow framing ignores the profound "
        "ways in which artistic culture sustains the social fabric and contributes "
        "to human flourishing.\n\n"
        "Cultural economists have long argued that the arts generate significant "
        "economic activity through tourism, creative industries, and urban regeneration. "
        "The Bilbao Effect — whereby the Guggenheim Museum transformed a declining "
        "industrial city into a vibrant cultural destination — offers a compelling "
        "counterpoint to the claim that art lacks practical value.\n\n"
        "Moreover, to reduce the question to economics is itself to concede too "
        "much to the philistine position. The arts cultivate empathy, challenge "
        "assumptions, and preserve collective memory in ways that no utilitarian "
        "calculus can adequately capture. Democratic societies have historically "
        "recognised this by supporting cultural institutions as public goods.\n\n"
        "That said, questions of prioritisation are legitimate. In times of austerity, "
        "the allocation of public funds demands justification. Arts organisations "
        "must engage more rigorously with questions of access and inclusion if "
        "public support is to remain defensible.\n\n"
        "Assessor Notes:\n"
        "Strong C2 vocabulary and generally sophisticated argument. However, the "
        "essay lacks the seamless flow and stylistic polish of top C2 writing. "
        "Some transitions feel slightly mechanical and the conclusion, while sensible, "
        "does not fully resolve the tensions raised. A confident but not exceptional "
        "C2 performance."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C2", "score": 3},
)

C2_ESSAY_2 = Document(
    page_content=(
        "CEFR Level C2 — Sample Essay 2 (Score: 5/5)\n\n"
        "Prompt: How far can silence be considered a form of communication?\n\n"
        "Student Essay:\n"
        "To ask whether silence communicates is, in one sense, to answer the question "
        "by asking it: the very hesitation before speech — the pause laden with "
        "implication — is already eloquent. Yet to claim that silence is merely "
        "a variant of language risks obscuring what is distinctive about it, "
        "for silence operates by a different logic than words.\n\n"
        "In interpersonal communication, silence is never semantically empty. "
        "The silence that follows an insult, the meditative quiet between old friends, "
        "the deliberate non-response to a question — each carries distinct affective "
        "weight. Linguists such as Deborah Tannen have demonstrated that conversational "
        "silence is profoundly culturally mediated: what reads as respectful "
        "attentiveness in one culture may signal sullen disengagement in another. "
        "This cultural variability is itself evidence that silence is interpreted "
        "rather than merely experienced — that it functions within a semiotic system.\n\n"
        "Yet silence also exceeds the semiotic. The silence of trauma — what "
        "psychiatrists call alexithymia, the inability to find words for inner "
        "states — points to a dimension of human experience that language fails "
        "to colonise entirely. Here silence is not a communicative act but its "
        "negation: the mark of an experience that resists articulation.\n\n"
        "What emerges, then, is a paradox: silence is both irreducibly communicative "
        "and irreducibly beyond communication. To speak of it at all — as I do now — "
        "is already to have broken it, to have imposed the grid of language upon "
        "something that, in its purest form, exceeds it. Perhaps that is why, "
        "as Wittgenstein famously observed, whereof one cannot speak, thereof "
        "one must be silent.\n\n"
        "Assessor Notes:\n"
        "Masterful C2 essay. The writing achieves a rare integration of analytical "
        "rigour and stylistic elegance. Vocabulary is precise and wide-ranging "
        "(semiotic, alexithymia, semantically, mediated, colonise). Complex syntactic "
        "structures — including inversions and appositional phrases — are deployed "
        "naturally. The argument is original, nuanced, and genuinely surprising. "
        "The concluding Wittgenstein reference is apt and not merely decorative. "
        "Indistinguishable from the work of an educated native speaker."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C2", "score": 5},
)

ALL_SAMPLE_ESSAYS = [
    A1_ESSAY_1, A1_ESSAY_2,
    A2_ESSAY_1, A2_ESSAY_2,
    B1_ESSAY_1, B1_ESSAY_2,
    B2_ESSAY_1, B2_ESSAY_2,
    C1_ESSAY_1, C1_ESSAY_2,
    C2_ESSAY_1, C2_ESSAY_2,
]
