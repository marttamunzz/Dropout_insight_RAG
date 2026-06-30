from typing import Any, cast

from rag.rag_recommender import generate_rag_recommendations


group_description = """
Cluster 0 formado por estudiantes con alto riesgo de abandono.
El grupo presenta bajo rendimiento académico y una reducción significativa
en las unidades curriculares aprobadas.
"""

counterfactuals_text = """
Los contrafactuales grupales indican que el riesgo de abandono disminuye
cuando aumentan las unidades curriculares aprobadas en el primer y segundo semestre,
y cuando las tasas de matrícula se encuentran al día.
"""


print("\nProbando Gemini...\n")

gemini_result = cast(
    dict[str, Any],
    generate_rag_recommendations(
        group_description=group_description,
        counterfactuals_text=counterfactuals_text,
        model_key="gemini",
        return_metadata=True,
        save_run=True,
    )
)

print(gemini_result["recommendation"])
print("\nTiempo Gemini:", gemini_result["response_time_seconds"])


print("\nProbando Groq...\n")

groq_result = cast(
    dict[str, Any],
    generate_rag_recommendations(
        group_description=group_description,
        counterfactuals_text=counterfactuals_text,
        model_key="groq",
        return_metadata=True,
        save_run=True,
    )
)

print(groq_result["recommendation"])
print("\nTiempo Groq:", groq_result["response_time_seconds"])