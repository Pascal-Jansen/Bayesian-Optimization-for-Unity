using System;
using System.Collections;
using BOforUnity;
using BOforUnity.Scripts;
using QuestionnaireToolkit.Scripts;
using UnityEngine;
using UnityEngine.UI;
using Application = UnityEngine.Application;

public class ColorGuesser : MonoBehaviour
{
    public Image image;

    public BoForUnityManager boManager;
    public QTQuestionnaireManager qtManager;

    [Header("BO Keys (empty: use the list position)")]
    public string redParameterKey = "Color-Red";
    public string greenParameterKey = "Color-Green";
    public string blueParameterKey = "Color-Blue";

    public void Awake()
    {
        StartCoroutine(GuessingRoutine());
    }

    private IEnumerator GuessingRoutine()
    {
        boManager = BoParameterReader.FindManager(boManager);

        // By key (else list position), normalized with each parameter's bounds.
        float r = 0.5f, g = 0.5f, b = 0.5f;
        if (!BoParameterReader.TryGetNormalized(boManager, redParameterKey, 0, out r) ||
            !BoParameterReader.TryGetNormalized(boManager, greenParameterKey, 1, out g) ||
            !BoParameterReader.TryGetNormalized(boManager, blueParameterKey, 2, out b))
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
}
