using System;
using System.Collections;
using System.Windows.Forms;
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
        boManager = GameObject.FindWithTag("BOforUnityManager").GetComponent<BoForUnityManager>();
        
        // Get new red, green, blue color parameter values from optimizer
        var parameterList = boManager.parameters;
        
        image.color = new Color(parameterList[0].value.Value, parameterList[1].value.Value, parameterList[2].value.Value);
        
        gameObject.GetComponent<ColorWheelMarker>().SetColor01(parameterList[0].value.Value, parameterList[1].value.Value, parameterList[2].value.Value);

        // Let the user experience the new color
        yield return new WaitForSecondsRealtime(1.5f);

        // call the questionnaire to receive the user feedback for the "Similarity" objective
        qtManager.StartQuestionnaire();
        
        yield return null;
    }

}
