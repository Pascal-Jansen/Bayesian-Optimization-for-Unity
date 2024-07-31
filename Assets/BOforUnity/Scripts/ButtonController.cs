using System;
using System.Collections;
using System.Collections.Generic;
using BOforUnity;
using TMPro;
using UnityEngine;

public class ButtonController : MonoBehaviour
{
    private BoForUnityManager _bomanager;

    public GameObject spinner;
    public GameObject button;
    public TMP_Text infoText;
    
    // Start is called before the first frame update
    void Start()
    {
        _bomanager = GameObject.FindWithTag("BOforUnityManager").GetComponent<BoForUnityManager>();
    }

    private void Update()
    {
        if (_bomanager && !_bomanager.optimizationRunning)
        {
            infoText.text = "You can now proceed!";
            spinner.SetActive(false);
            button.SetActive(true);
        }
    }

    public void Next()
    {
        _bomanager.ButtonNextIteration();
    }
}
