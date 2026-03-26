"""CEFR level descriptors for writing proficiency.

Each descriptor explains what a learner at that CEFR level can do in writing.
Retrieved during the first retrieval phase to help the LLM ground its
scoring against official standards.

6 documents: one per CEFR level (A1-C2).
"""

from langchain_core.documents import Document

A1_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level A1 — Writing Proficiency Descriptor\n\n"
        "Can write simple isolated phrases and sentences. Can fill in forms with "
        "personal details (name, nationality, address). Can write a short simple "
        "postcard (holiday greetings). Can write simple phrases and sentences about "
        "themselves and imaginary people, where they live and what they do. Writing "
        "is limited to isolated words and formulaic expressions with frequent "
        "spelling and grammar errors."
    ),
    metadata={"type": "standard", "cefr_level": "A1"},
)

A2_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level A2 — Writing Proficiency Descriptor\n\n"
        "Can write short, simple notes and messages. Can write a very simple "
        "personal letter (thanking someone). Can write about everyday aspects of "
        "their environment (people, places, job, school) in linked sentences. "
        "Can describe plans and arrangements, habits and routines, past activities "
        "and personal experiences. Uses simple vocabulary and basic sentence patterns. "
        "Errors are common but meaning is usually clear."
    ),
    metadata={"type": "standard", "cefr_level": "A2"},
)

B1_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level B1 — Writing Proficiency Descriptor\n\n"
        "Can write straightforward connected text on familiar topics. Can write "
        "personal letters describing experiences and impressions. Can write essays "
        "or reports presenting information and giving reasons for or against a point "
        "of view. Uses a reasonable range of vocabulary and grammar structures. "
        "Can link ideas using basic connectors (because, however, although). "
        "Errors occur but rarely impede communication."
    ),
    metadata={"type": "standard", "cefr_level": "B1"},
)

B2_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level B2 — Writing Proficiency Descriptor\n\n"
        "Can write clear, detailed text on a wide range of subjects. Can write an "
        "essay or report passing on information or giving reasons for or against a "
        "particular point of view. Can write letters highlighting the personal "
        "significance of events and experiences. Uses a wide range of vocabulary "
        "and complex sentence structures. Good control of grammar with occasional "
        "errors in less common structures. Can develop arguments systematically."
    ),
    metadata={"type": "standard", "cefr_level": "B2"},
)

C1_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level C1 — Writing Proficiency Descriptor\n\n"
        "Can write well-structured, clear text about complex subjects. Can select "
        "appropriate style for the reader in mind. Can write detailed expositions, "
        "proposals, and reviews. Uses a broad range of vocabulary with precision and "
        "flexibility. Demonstrates consistent grammatical accuracy of complex "
        "language. Can use language effectively for social, academic, and professional "
        "purposes with only occasional minor errors."
    ),
    metadata={"type": "standard", "cefr_level": "C1"},
)

C2_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level C2 — Writing Proficiency Descriptor\n\n"
        "Can write clear, smoothly flowing text in an appropriate style. Can write "
        "complex letters, reports, or articles presenting a case with effective "
        "logical structure. Can write summaries and reviews of professional or "
        "literary works. Demonstrates mastery of a very broad vocabulary including "
        "idiomatic expressions and colloquialisms. Virtually no errors. Writing "
        "is indistinguishable from that of an educated native speaker."
    ),
    metadata={"type": "standard", "cefr_level": "C2"},
)

ALL_STANDARDS = [
    A1_DESCRIPTOR, A2_DESCRIPTOR, B1_DESCRIPTOR,
    B2_DESCRIPTOR, C1_DESCRIPTOR, C2_DESCRIPTOR,
]
