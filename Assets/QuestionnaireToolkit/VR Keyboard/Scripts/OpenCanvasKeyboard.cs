using System;
using QuestionnaireToolkit.Scripts;
using UnityEngine;

namespace QuestionnaireToolkit.VR_Keyboard.Scripts
{
	public class OpenCanvasKeyboard : MonoBehaviour 
	{
		// Canvas to open keyboard under
		public GameObject CanvasKeyboardObject;

		// Optional: Input Object to receive text 
		public GameObject inputObject;

		public RectTransform textInputParent;

		private QTQuestionnaireManager questionnaireManager;
		
		private void Start()
		{
			try
			{
				var q = GameObject.FindWithTag("QTManager").GetComponent<QTManager>();
				questionnaireManager = q.FindParentWithTag(gameObject, "QTQuestionnaireManager").GetComponent<QTQuestionnaireManager>();
			}
			catch (Exception)
			{
				questionnaireManager = GameObject.FindWithTag("QTQuestionnaireManager").GetComponent<QTQuestionnaireManager>();
			}
		}

		public void OpenKeyboard()
		{
			if (questionnaireManager.displayMode == QTQuestionnaireManager.DisplayMode.VR)
				CanvasKeyboard.Open(CanvasKeyboardObject, inputObject != null ? inputObject : gameObject, textInputParent);
		}

		public void CloseKeyboard() 
		{		
			CanvasKeyboard.Close ();
		}
	}
}