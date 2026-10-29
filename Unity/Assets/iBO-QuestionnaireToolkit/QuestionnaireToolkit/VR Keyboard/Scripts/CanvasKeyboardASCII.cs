using UnityEngine;

namespace QuestionnaireToolkit.VR_Keyboard.Scripts
{
	public class CanvasKeyboardASCII : MonoBehaviour 
	{
		public CanvasKeyboard canvasKeyboard;
		private bool ShiftDown = false;
		private bool AltDown = false;
		public GameObject alphaBoardUnsfhifted;
		public GameObject alphaBoardSfhifted;
		public GameObject numberBoardUnshifted;
		public GameObject numberBoardShifted;

		void Awake() 
		{
			Refresh();
		}
		
		void Refresh()
		{
			// Show the current board
			alphaBoardUnsfhifted.SetActive(!AltDown && !ShiftDown);
			alphaBoardSfhifted.SetActive(!AltDown && ShiftDown);
			numberBoardUnshifted.SetActive(AltDown && !ShiftDown);
			numberBoardShifted.SetActive(AltDown && ShiftDown);
		}

		public void OnKeyDown(GameObject kb)
		{
			if (kb.name == "DONE")
			{
				if (canvasKeyboard != null)
				{
					canvasKeyboard.CloseKeyboard();
				}
			}
			else if (kb.name == "ALT")
			{
				AltDown = !AltDown;
				ShiftDown = false;
				Refresh ();
			}
			else if (kb.name == "SHIFT")
			{
				ShiftDown = !ShiftDown;
				Refresh ();
			}
			else
			{
				if (canvasKeyboard != null)
				{
					string s;
					if (kb.name == "BACKSPACE")
					{
						s = "\x08";
					}
					else
					{
						s = kb.name;
					}
					canvasKeyboard.SendKeyString(s);
				}
			}
			
		}

	}
}