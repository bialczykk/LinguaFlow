"""CEFR assessment rubrics for scoring student writing.

Each rubric describes what scores 1-5 look like for a specific assessment
dimension at a specific CEFR level band. These are stored in the Chroma
vector store and retrieved during the criteria_scoring phase.

4 dimensions x 3 level bands = 12 rubric documents.
"""

from langchain_core.documents import Document

# -- Grammar & Accuracy Rubrics --

GRAMMAR_A1_A2 = Document(
    page_content=(
        "Grammar & Accuracy Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: Cannot form basic sentences. Frequent errors in word order, "
        "verb forms, and basic structures that prevent comprehension.\n"
        "Score 2: Attempts simple sentences but with frequent errors in subject-verb "
        "agreement, tense use, and articles. Meaning is often unclear.\n"
        "Score 3: Can produce simple sentences with some accuracy. Common errors in "
        "tenses and articles but meaning is generally clear. Uses present simple "
        "and past simple with moderate accuracy.\n"
        "Score 4: Good control of basic structures. Errors are infrequent and do not "
        "impede understanding. Beginning to use some complex structures.\n"
        "Score 5: Excellent control of A1-A2 grammar. Accurate use of present, past, "
        "and future tenses. Very few errors in basic structures."
    ),
    metadata={"type": "rubric", "dimension": "grammar", "level_band": "A1-A2"},
)

GRAMMAR_B1_B2 = Document(
    page_content=(
        "Grammar & Accuracy Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Struggles with intermediate structures. Frequent errors in "
        "conditionals, passive voice, and complex tenses impede meaning.\n"
        "Score 2: Attempts complex structures but with regular errors. Inconsistent "
        "use of reported speech, relative clauses, and modal verbs.\n"
        "Score 3: Reasonable control of intermediate grammar. Uses conditionals, "
        "passive voice, and relative clauses with some errors. Meaning is clear "
        "despite occasional mistakes.\n"
        "Score 4: Good range of structures used accurately. Occasional slips in "
        "complex areas but these do not impede communication.\n"
        "Score 5: Confident and accurate use of B1-B2 grammar. Complex sentences, "
        "conditionals, and passive constructions are well-formed."
    ),
    metadata={"type": "rubric", "dimension": "grammar", "level_band": "B1-B2"},
)

GRAMMAR_C1_C2 = Document(
    page_content=(
        "Grammar & Accuracy Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Errors in advanced structures undermine otherwise competent writing. "
        "Struggles with nuanced tense choices and complex subordination.\n"
        "Score 2: Some control of advanced grammar but noticeable errors in "
        "subjunctive mood, inversion, and cleft sentences.\n"
        "Score 3: Good control of advanced grammar. Uses a wide range of structures "
        "with occasional errors in the most complex constructions.\n"
        "Score 4: Consistently accurate use of advanced grammar. Minor slips "
        "are rare and stylistic rather than structural.\n"
        "Score 5: Near-native accuracy. Masterful use of nuanced grammar including "
        "inversions, cleft sentences, and sophisticated subordination."
    ),
    metadata={"type": "rubric", "dimension": "grammar", "level_band": "C1-C2"},
)

VOCABULARY_A1_A2 = Document(
    page_content=(
        "Vocabulary Range & Precision Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: Extremely limited vocabulary. Relies on a few memorized words "
        "and phrases. Cannot express basic ideas without significant gaps.\n"
        "Score 2: Basic vocabulary for familiar topics but frequent gaps. Relies "
        "heavily on repetition and simple words.\n"
        "Score 3: Adequate vocabulary for everyday topics. Can describe basic "
        "situations using common words. Some word choice errors but meaning is clear.\n"
        "Score 4: Good range of basic vocabulary. Appropriate word choices for "
        "familiar contexts. Beginning to use some less common words.\n"
        "Score 5: Strong basic vocabulary. Precise word choices for everyday topics. "
        "Shows awareness of collocations at the basic level."
    ),
    metadata={"type": "rubric", "dimension": "vocabulary", "level_band": "A1-A2"},
)

VOCABULARY_B1_B2 = Document(
    page_content=(
        "Vocabulary Range & Precision Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Limited vocabulary for the level. Over-relies on basic words "
        "when intermediate vocabulary is expected. Frequent imprecision.\n"
        "Score 2: Some intermediate vocabulary but used inconsistently. "
        "Paraphrases when specific terms would be more appropriate.\n"
        "Score 3: Reasonable vocabulary range. Uses topic-specific terms and "
        "some idiomatic expressions. Occasional imprecision but generally effective.\n"
        "Score 4: Good intermediate vocabulary with appropriate use of less common "
        "words. Shows awareness of register and collocation.\n"
        "Score 5: Wide vocabulary range for the level. Precise, varied word choices "
        "including idiomatic expressions and topic-specific terminology."
    ),
    metadata={"type": "rubric", "dimension": "vocabulary", "level_band": "B1-B2"},
)

VOCABULARY_C1_C2 = Document(
    page_content=(
        "Vocabulary Range & Precision Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Vocabulary does not match the expected advanced level. "
        "Relies on intermediate-level words when sophisticated choices are needed.\n"
        "Score 2: Some advanced vocabulary but lacks precision. Occasional "
        "misuse of nuanced terms or inappropriate register.\n"
        "Score 3: Good advanced vocabulary. Uses sophisticated terms, academic "
        "language, and idiomatic expressions with reasonable precision.\n"
        "Score 4: Wide-ranging, precise vocabulary. Effective use of nuance, "
        "connotation, and register-appropriate language.\n"
        "Score 5: Near-native vocabulary mastery. Exceptional precision and range "
        "including rare words, technical terms, and subtle distinctions."
    ),
    metadata={"type": "rubric", "dimension": "vocabulary", "level_band": "C1-C2"},
)

COHERENCE_A1_A2 = Document(
    page_content=(
        "Coherence & Organization Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: No discernible organization. Isolated words or phrases "
        "without logical connection.\n"
        "Score 2: Minimal organization. Ideas are listed but not connected. "
        "No use of linking words.\n"
        "Score 3: Basic organization with simple linking words (and, but, because). "
        "Ideas follow a simple logical order. Paragraphing may be absent.\n"
        "Score 4: Clear basic organization. Uses simple connectors effectively. "
        "Ideas are logically sequenced with some paragraphing.\n"
        "Score 5: Well-organized for the level. Clear introduction and conclusion. "
        "Effective use of basic connectors and logical sequencing."
    ),
    metadata={"type": "rubric", "dimension": "coherence", "level_band": "A1-A2"},
)

COHERENCE_B1_B2 = Document(
    page_content=(
        "Coherence & Organization Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Poor organization for the level. Ideas are disjointed and "
        "transitions are missing or ineffective.\n"
        "Score 2: Some organization but inconsistent. Paragraphs exist but "
        "topic sentences and transitions are weak.\n"
        "Score 3: Reasonable organization. Clear paragraphs with topic sentences. "
        "Uses a range of linking devices (however, therefore, in addition) "
        "with some effectiveness.\n"
        "Score 4: Good organization with clear logical flow. Effective use of "
        "cohesive devices. Ideas are well-developed within paragraphs.\n"
        "Score 5: Excellent organization. Skillful use of transitions and "
        "cohesive devices. Strong paragraph structure with clear progression."
    ),
    metadata={"type": "rubric", "dimension": "coherence", "level_band": "B1-B2"},
)

COHERENCE_C1_C2 = Document(
    page_content=(
        "Coherence & Organization Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Organization does not match the expected advanced level. "
        "Lacks sophisticated structuring despite advanced language use.\n"
        "Score 2: Some advanced organizational features but inconsistent. "
        "Transitions between complex ideas are sometimes abrupt.\n"
        "Score 3: Good advanced organization. Effective use of discourse markers "
        "and referencing. Complex ideas are clearly structured.\n"
        "Score 4: Sophisticated organization. Skillful management of complex "
        "information with seamless transitions and cohesive referencing.\n"
        "Score 5: Masterful organization. Text flows naturally with implicit "
        "and explicit coherence devices. Complex arguments are elegantly structured."
    ),
    metadata={"type": "rubric", "dimension": "coherence", "level_band": "C1-C2"},
)

TASK_ACHIEVEMENT_A1_A2 = Document(
    page_content=(
        "Task Achievement Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: Does not address the task. Response is off-topic or "
        "incomprehensible.\n"
        "Score 2: Partially addresses the task. Some relevant content but "
        "significant parts of the prompt are ignored.\n"
        "Score 3: Addresses the main points of the task. Response is relevant "
        "though may lack detail or completeness.\n"
        "Score 4: Fully addresses the task with adequate detail. All parts "
        "of the prompt are covered.\n"
        "Score 5: Thoroughly addresses the task with good detail and development. "
        "Goes beyond the minimum requirements."
    ),
    metadata={"type": "rubric", "dimension": "task_achievement", "level_band": "A1-A2"},
)

TASK_ACHIEVEMENT_B1_B2 = Document(
    page_content=(
        "Task Achievement Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Fails to address the task requirements. Response is superficial "
        "or largely irrelevant.\n"
        "Score 2: Partially addresses the task. Ideas are underdeveloped and "
        "some key points are missing.\n"
        "Score 3: Adequately addresses the task. Main ideas are developed with "
        "supporting details. Some aspects could be more thorough.\n"
        "Score 4: Fully addresses the task with well-developed ideas. Supporting "
        "evidence and examples are relevant and effective.\n"
        "Score 5: Comprehensively addresses the task. Ideas are fully developed "
        "with compelling evidence. Demonstrates critical thinking."
    ),
    metadata={"type": "rubric", "dimension": "task_achievement", "level_band": "B1-B2"},
)

TASK_ACHIEVEMENT_C1_C2 = Document(
    page_content=(
        "Task Achievement Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Response does not meet advanced-level expectations for task "
        "completion. Analysis is superficial.\n"
        "Score 2: Some depth but insufficient for the level. Arguments lack "
        "nuance and critical analysis.\n"
        "Score 3: Good task achievement. Arguments are well-developed with "
        "supporting evidence. Shows analytical thinking.\n"
        "Score 4: Thorough task achievement. Sophisticated analysis with "
        "well-supported arguments and critical evaluation.\n"
        "Score 5: Exceptional task achievement. Insightful, nuanced response "
        "with compelling arguments and original thinking."
    ),
    metadata={"type": "rubric", "dimension": "task_achievement", "level_band": "C1-C2"},
)

ALL_RUBRICS = [
    GRAMMAR_A1_A2, GRAMMAR_B1_B2, GRAMMAR_C1_C2,
    VOCABULARY_A1_A2, VOCABULARY_B1_B2, VOCABULARY_C1_C2,
    COHERENCE_A1_A2, COHERENCE_B1_B2, COHERENCE_C1_C2,
    TASK_ACHIEVEMENT_A1_A2, TASK_ACHIEVEMENT_B1_B2, TASK_ACHIEVEMENT_C1_C2,
]
