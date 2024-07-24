using System.Collections;
using System.Reflection;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
#if HTC_Vive
 using HTC.UnityPlugin.Pointer3D;
 using HTC.UnityPlugin.Vive;
#endif
#if Oculus

#endif

namespace QuestionnaireToolkit.VR_Keyboard.Scripts
{
	public class CanvasKeyboard : MonoBehaviour
	{
		private ScrollRect scrollRect;
		
		#region CanvasKeyboard Instantiation

		public enum CanvasKeyboardType
		{
			ASCIICapable
		}
		
		public static CanvasKeyboard Open(GameObject parent = null, GameObject inputObject = null, RectTransform textInputparent = null, CanvasKeyboardType keyboardType = CanvasKeyboardType.ASCIICapable)
		{
			// Don't open the keyboard if it is already open for the current input object
			CanvasKeyboard keyboard = GameObject.FindObjectOfType<CanvasKeyboard>();
			if (keyboard == null || (keyboard != null && keyboard.inputObject != inputObject))
			{
				Close();
				keyboard = Instantiate<CanvasKeyboard>(Resources.Load<CanvasKeyboard>("CanvasKeyboard"), textInputparent.transform, false);
				//keyboard.transform.SetParent(parent.transform, false);
				
				keyboard.GetComponent<RectTransform>().anchoredPosition3D = new Vector3(0, (-1) * keyboard.GetComponent<RectTransform>().sizeDelta.y + 10, 0);
				keyboard.inputObject = inputObject;

				if (textInputparent.CompareTag("QTOptionOther"))
				{
					keyboard.scrollRect = textInputparent.parent.parent.parent.parent.parent.GetComponent<ScrollRect>();	
				}
				else
				{
					keyboard.scrollRect = textInputparent.parent.parent.parent.GetComponent<ScrollRect>();
				}
				keyboard.scrollRect.enabled = false;

#if HTC_Vive
				keyboard.GetComponent<GraphicRaycaster>().enabled = false;
				keyboard.gameObject.AddComponent<CanvasRaycastTarget>();
#endif
#if Oculus
				keyboard.GetComponent<GraphicRaycaster>().enabled = false;
				keyboard.gameObject.AddComponent<OVRRaycaster>().blockingObjects = OVRRaycaster.BlockingObjects.All;
                keyboard.GetComponent<Canvas>().worldCamera = Camera.main;
				if (textInputparent.CompareTag("QTOptionOther"))
                {
	                textInputparent.parent.parent.parent.parent.parent.parent.GetComponent<OVRRaycaster>().enabled = false;	
                }
                else
                {
	                textInputparent.parent.parent.parent.parent.GetComponent<OVRRaycaster>().enabled = false;
                }
#endif
			}
			return keyboard;
		}
		
		public static void Close()
		{
			CanvasKeyboard[] kbs = GameObject.FindObjectsOfType<CanvasKeyboard>();
			foreach (CanvasKeyboard kb in kbs)
			{
				kb.CloseKeyboard();
			}
		}
		
		public static bool IsOpen 
		{
			get
			{
				return GameObject.FindObjectsOfType<CanvasKeyboard>().Length != 0;
			}
		}

		#endregion

		public GameObject inputObject;

		public string text 
		{
			get
			{
				if (inputObject != null) 
				{
					Component[] components = inputObject.GetComponents(typeof(Component));
					foreach (Component component in components)
					{
						PropertyInfo prop = component.GetType().GetProperty("text", BindingFlags.Instance | BindingFlags.Public);
						if (prop != null)
						{
							return prop.GetValue(component, null)  as string;;
						}
					}
					return inputObject.name;
				}
				return "";
			}
			
			set 
			{
				if (inputObject != null) 
				{
					Component[] components = inputObject.GetComponents(typeof(Component));
					foreach (Component component in components)
					{
						PropertyInfo prop = component.GetType().GetProperty("text", BindingFlags.Instance | BindingFlags.Public);
						if (prop != null)
						{
							prop.SetValue(component, value, null);
							return;
						}
					}
					inputObject.name = value;
				}
			}
		}

		#region Keyboard Receiving Input

		public void SendKeyString(string keyString)
		{
			if (keyString.Length == 1 && keyString[0] == 8/*ASCII.Backspace*/)
			{
				if (text.Length > 0)
				{
					text = text.Remove(text.Length - 1); 
				}
			}
			else
			{
				text += keyString;
			}

			// Workaround: Restore focus to input fields (because Unity UI buttons always steal focus)
			ReactivateInputField(inputObject.GetComponent<InputField>());

		}

		public void CloseKeyboard()
		{
			scrollRect.enabled = true;
#if Oculus
			if (transform.parent.CompareTag("QTOptionOther"))
			{
				transform.parent.parent.parent.parent.parent.parent.parent.GetComponent<OVRRaycaster>().enabled = true;	
			}
			else
			{
				transform.parent.parent.parent.parent.parent.GetComponent<OVRRaycaster>().enabled = true;
			}
#endif
			Destroy(gameObject);
		}

		#endregion


		#region Steal Focus Workaround

		void ReactivateInputField(InputField inputField)
		{
			if (inputField != null)
			{
				StartCoroutine(ActivateInputFieldWithoutSelection(inputField));
			}
		}

		IEnumerator ActivateInputFieldWithoutSelection(InputField inputField)
		{
			inputField.ActivateInputField();

			// wait for the activation to occur in a lateupdate
			yield return new WaitForEndOfFrame();

			// make sure we're still the active ui
			if (EventSystem.current.currentSelectedGameObject == inputField.gameObject)
			{
				// To remove hilight we'll just show the caret at the end of the line
				inputField.MoveTextEnd(false);
			}
		}

		#endregion

	}
}