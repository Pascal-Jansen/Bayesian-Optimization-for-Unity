using System;
using System.Collections;
using BOforUnity;
using QuestionnaireToolkit.Scripts;
using UnityEngine;
using UnityEngine.UI;
using Application = UnityEngine.Application;

public class ColorGuesser : MonoBehaviour
{
    public Image image;

    public BoForUnityManager boManager;
    public QTQuestionnaireManager qtManager;

    public void Awake()
    {
        StartCoroutine(GuessingRoutine());
    }

    private IEnumerator GuessingRoutine()
    {
        boManager = FindObjectOfType<BoForUnityManager>();

        float r = 0.5f, g = 0.5f, b = 0.5f;
        if (!TryGetNormalizedParameterByIndex(0, out r) ||
            !TryGetNormalizedParameterByIndex(1, out g) ||
            !TryGetNormalizedParameterByIndex(2, out b))
        {
            Debug.LogWarning(
                "ColorGuesser: Could not read three valid BO parameters for RGB. " +
                "Using neutral fallback color (0.5, 0.5, 0.5)."
            );
        }

        if (image != null)
        {
            image.color = new Color(r, g, b);
        }

        var marker = gameObject.GetComponent<ColorWheelMarker>();
        if (marker != null)
        {
            marker.SetColor01(r, g, b);
        }

        // Let the user experience the new color
        yield return new WaitForSecondsRealtime(1.5f);

        // call the questionnaire to receive the user feedback for the "Similarity" objective
        if (qtManager != null)
        {
            qtManager.StartQuestionnaire();
        }
        else
        {
            Debug.LogWarning("ColorGuesser: QTQuestionnaireManager reference is missing.");
        }
        
        yield return null;
    }

    private bool TryGetNormalizedParameterByIndex(int validIndex, out float normalizedValue)
    {
        normalizedValue = 0.5f;
        if (boManager == null || boManager.parameters == null || validIndex < 0)
            return false;

        int seenValid = 0;
        for (int i = 0; i < boManager.parameters.Count; i++)
        {
            var parameter = boManager.parameters[i];
            if (parameter == null || parameter.value == null || string.IsNullOrWhiteSpace(parameter.key))
                continue;

            if (seenValid == validIndex)
            {
                normalizedValue = Mathf.Clamp01(parameter.value.Value);
                return true;
            }
            seenValid++;
        }

        return false;
    }

}
