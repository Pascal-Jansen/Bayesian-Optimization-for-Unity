using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace Ludiq.Controls.Editor
{
	/// <summary>
	/// Utility class to display complex editor popups.
	/// </summary>
	public static partial class DropdownGUI<T>
	{
		// Note: a selected option outside the allowed options
		// will stay selected, because the check isn't done at every frame
		// for performance reasons. Therefore, if the options list changes, 
		// it's up to the user to manually check if the option is in range.

		/// <summary>
		/// The unique control ID of the currently active popup.
		/// </summary>
		private static int activePopupControlID = -1;

		/// <summary>
		/// Whether the dropdown selection has changed.
		/// </summary>
		private static bool activeDropdownChanged = false;

		/// <summary>
		/// A method to build a list of dropdown options.
		/// </summary>
		public delegate IEnumerable<DropdownOption<T>> GetOptionsCallback();
		
		// Shorthand
		private static bool Equal(T a, T b) 
		{
			return EqualityComparer<T>.Default.Equals(a, b);
		}

		// Get the internal last control ID via reflection
		private static FieldInfo lastControlIDField;
		private static int GetLastControlID()
		{
			if (lastControlIDField == null)
			{
				lastControlIDField = typeof(EditorGUIUtility).GetField("s_LastControlID", BindingFlags.Static | BindingFlags.NonPublic);
			}

			return (int)lastControlIDField.GetValue(null);
		}

		#region Single

		/// <summary>
		/// The selected value in the currently active single-option popup dropdown.
		/// </summary>
		private static T activeSingleDropdownValue;

		/// <summary>
		/// A method to call when a value has been selected in a single-option dropdown.
		/// </summary>
		public delegate void SingleCallback(T value);

		/// <summary>
		/// Displays a single-option dropdown.
		/// </summary>
		public static void DropdownSingle
		(
			Vector2 position,
			SingleCallback callback,
			IEnumerable<DropdownOption<T>> options,
			DropdownOption<T> selectedOption,
			DropdownOption<T> noneOption
		)
		{
			var hasMultipleDifferentValues = EditorGUI.showMixedValue;

            ICollection<DropdownOption<T>> optionsCache = null;

			bool hasOptions;

			if (options != null)
			{
				optionsCache = options.CacheToCollection();
				hasOptions = optionsCache.Count > 0;
			}
			else
			{
				hasOptions = false;
			}

			GenericMenu menu = new GenericMenu();
			GenericMenu.MenuFunction2 menuCallback = (o) => { callback((T)o); };

			if (noneOption != null)
			{
				bool on = !hasMultipleDifferentValues && (selectedOption == null || Equal(selectedOption.value, noneOption.value));

				menu.AddItem(new GUIContent(noneOption.label), on, menuCallback, noneOption.value);
			}

			if (noneOption != null && hasOptions)
			{
				menu.AddSeparator(string.Empty);
			}

			if (hasOptions)
			{
				foreach (var option in optionsCache)
				{
					bool on = !hasMultipleDifferentValues && (selectedOption != null && Equal(selectedOption.value, option.value));

					menu.AddItem(new GUIContent(option.label), on, menuCallback, option.value);
				}
			}

			menu.DropDown(new Rect(position, Vector2.zero));
		}

		/// <summary>
		/// Displays a single-option popup.
		/// </summary>
		public static T PopupSingle
		(
			Rect position,
			IEnumerable<DropdownOption<T>> options,
			T selectedValue,
			DropdownOption<T> noneOption,
			GUIContent label = null,
			GUIStyle style = null
		)
		{
			var selectedOption = options.FirstOrDefault(o => Equal(o.value, selectedValue));

			if (selectedOption == null)
			{
				selectedOption = new DropdownOption<T>(selectedValue);
			}

			return PopupSingle
			(
				position,
				options,
				selectedOption,
				noneOption,
				label,
				style
			);
		}

		/// <summary>
		/// Displays a single-option popup with a custom selected option label.
		/// </summary>
		public static T PopupSingle
		(
			Rect position,
			IEnumerable<DropdownOption<T>> options,
			DropdownOption<T> selectedOption,
			DropdownOption<T> noneOption,
			GUIContent label = null,
			GUIStyle style = null
		)
		{
			return PopupSingle
			(
				position,
				() => options,
				selectedOption,
				noneOption,
				label,
				style
			);
		}

		/// <summary>
		/// Displays a single-option popup where the option list is only built when the dropdown is shown.
		/// Useful when building the options list is very demanding (e.g. reflection).
		/// </summary>
		public static T PopupSingle
		(
			Rect position,
			GetOptionsCallback getOptions,
			T selectedValue,
			DropdownOption<T> noneOption,
			GUIContent label = null,
			GUIStyle style = null
		)
		{
			var selectedOption = new DropdownOption<T>(selectedValue);

			return PopupSingle
			(
				position,
				getOptions,
				selectedOption,
				noneOption,
				label,
				style
			);
		}

		/// <summary>
		/// Displays a single-option popup where the option list is only built when the dropdown is shown.
		/// Useful when building the options list is very demanding (e.g. reflection).
		/// </summary>
		public static T PopupSingle
		(
			Rect position,
			GetOptionsCallback getOptions,
			DropdownOption<T> selectedOption,
			DropdownOption<T> noneOption,
			GUIContent label = null,
			GUIStyle style = null
		)
		{
			var hasMultipleDifferentValues = EditorGUI.showMixedValue;

			// Determine the label text if no override is specified
			if (label == null)
			{
				string text;

				if (hasMultipleDifferentValues)
				{
					text = "\u2014"; // Em Dash
				}
				else if (selectedOption == null)
				{
					if (noneOption != null)
					{
						text = noneOption.label;
					}
					else
					{
						text = string.Empty;
					}
				}
				else
				{
					text = selectedOption.label;
				}

				label = new GUIContent(text);
			}

			// Apply the popup style is no override is specified
			if (style == null)
			{
				style = EditorStyles.popup;
			}

			// Render a button and get its control ID
			var popupClicked = GUI.Button(position, label, style);
			var popupControlID = GetLastControlID();

			if (popupClicked)
			{
				// Cancel button click
				GUI.changed = false; 

				// Assign the active control ID
				activePopupControlID = popupControlID;

				// Display the dropdown
				DropdownSingle
				(
					new Vector2(position.xMin, position.yMax),
					(value) =>
					{
						activeSingleDropdownValue = value;
						activeDropdownChanged = true;
					},
					getOptions(),
					selectedOption,
					noneOption
				);
			}

			if (popupControlID == activePopupControlID && activeDropdownChanged) // Selected option changed
			{
				// TODO: Use EditorWindow.SendEvent like EditorGUI.PopupCallbackInfo does.
				// Otherwise, there seems to be a 1-frame delay in update.
				GUI.changed = true;
				activePopupControlID = -1;
				activeDropdownChanged = false;
				return activeSingleDropdownValue;
			}
			else if (selectedOption == null) // Selected option is null
			{
				if (noneOption != null)
				{
					return noneOption.value;
				}
				else
				{
					return default(T);
				}
			}
			else
			{
				return selectedOption.value;
			}
		}

		#endregion

		#region Multiple
		
		/// <summary>
		/// The selected values in the currently active multiple-options popup dropdown.
		/// </summary>
		private static HashSet<T> activeMultipleDropdownValues;

		/// <summary>
		/// A method to call when values have changed in a multiple-options dropdown.
		/// </summary>
		public delegate void MultipleCallback(HashSet<T> value);

		/// <summary>
		/// Displays a multiple-options dropdown.
		/// </summary>
		public static void DropdownMultiple
		(
			Vector2 position,
			MultipleCallback callback,
			IEnumerable<DropdownOption<T>> options,
			HashSet<T> selectedOptions,
			bool showNothingEverything = true
		)
		{
			var hasMultipleDifferentValues = EditorGUI.showMixedValue;

			ICollection<DropdownOption<T>> optionsCache = null;

			bool hasOptions;

			if (options != null)
			{
				optionsCache = options.CacheToCollection();
				hasOptions = optionsCache.Count > 0;
			}
			else
			{
				hasOptions = false;
			}

			var selectedOptionsCopy = selectedOptions.ToHashSet();

			// Remove options outside range
			selectedOptionsCopy.RemoveWhere(so => !optionsCache.Any(o => Equal(o.value, so)));

			GenericMenu menu = new GenericMenu();

			// The callback when a normal option has been selected
			GenericMenu.MenuFunction2 switchCallback = (o) =>
			{
				var switchOption = (T)o;
				
				if (selectedOptionsCopy.Contains(switchOption))
				{
					selectedOptionsCopy.Remove(switchOption);
				}
				else
				{
					selectedOptionsCopy.Add(switchOption);
				}

				callback(selectedOptionsCopy); // Force copy
			};

			// The callback when the special "Nothing" option has been selected
			GenericMenu.MenuFunction nothingCallback = () =>
			{
				callback(new HashSet<T>());
			};

			// The callback when the special "Everything" option has been selected
			GenericMenu.MenuFunction everythingCallback = () =>
			{
				callback(optionsCache.Select((o) => o.value).ToHashSet());
			};

			// Add the special "Nothing" / "Everything" options
			if (showNothingEverything)
			{
				menu.AddItem
				(
					new GUIContent("Nothing"), 
					!hasMultipleDifferentValues && selectedOptionsCopy.Count == 0, 
					nothingCallback
				);

				if (hasOptions)
				{
					menu.AddItem
					(
						new GUIContent("Everything"),
						!hasMultipleDifferentValues && selectedOptionsCopy.Count == optionsCache.Count && Enumerable.SequenceEqual(selectedOptionsCopy.OrderBy(t => t), optionsCache.Select(o => o.value).OrderBy(t => t)),
						everythingCallback
					);
				}
			}

			// Add a separator (not in Unity default, but pretty)
			if (showNothingEverything && hasOptions)
			{
				menu.AddSeparator(string.Empty);
			}

			// Add the normal options
			if (hasOptions)
			{
				foreach (var option in optionsCache)
				{
					menu.AddItem
					(
						new GUIContent(option.label),
						!hasMultipleDifferentValues && (selectedOptionsCopy.Any(selectedOption => Equal(selectedOption, option.value))), 
						switchCallback, 
						option.value
					);
				}
			}

			// Show the dropdown
			menu.DropDown(new Rect(position, Vector2.zero));
		}

		/// <summary>
		/// Displays a multiple-options popup.
		/// </summary>
		public static HashSet<T> PopupMultiple
		(
			Rect position,
			IEnumerable<DropdownOption<T>> options,
			HashSet<T> selectedOptions,
			bool showNothingEverything = true,
			GUIContent label = null,
			GUIStyle style = null
		)
		{
			return PopupMultiple
			(
				position,
				() => options,
				selectedOptions,
				showNothingEverything,
				label,
				style
			);
		}

		/// <summary>
		/// Displays a multiple-options popup where the option list is only built when the dropdown is shown.
		/// Useful when building the options list is very demanding (e.g. reflection).
		/// Note that not specifying a label will force the options list to be built at every repaint.
		/// </summary>
		public static HashSet<T> PopupMultiple
		(
			Rect position,
			GetOptionsCallback getOptions,
			HashSet<T> selectedOptions,
			bool showNothingEverything = true,
			GUIContent label = null,
			GUIStyle style = null
		)
		{
			var hasMultipleDifferentValues = EditorGUI.showMixedValue;

			IEnumerable<DropdownOption<T>> options = null;

			// Determine the label text if no override is specified
			if (label == null)
			{
				options = getOptions();
				
				string text;

				if (hasMultipleDifferentValues)
				{
					text = "\u2014"; // Em Dash
				}
				else
				{
					var selectedOptionsCount = selectedOptions.Count();
					var optionsCount = options.Count();

					if (selectedOptionsCount == 0)
					{
						text = "Nothing";
					}
					else if (selectedOptionsCount == 1)
					{
						text = options.First(o => Equal(o.value, selectedOptions.First())).label;
					}
					else if (selectedOptionsCount == optionsCount)
					{
						text = "Everything";
					}
					else
					{
						text = "(Mixed)";
					}
				}

				label = new GUIContent(text);
			}

			// Apply the popup style if no override is specified
			if (style == null)
			{
				style = EditorStyles.popup;
			}

			// Render a button and get its control ID
			var popupClicked = GUI.Button(position, label, style);
			var popupControlID = GetLastControlID();

			if (popupClicked)
			{
				// Cancel button click
				GUI.changed = false;

				// Assign the currently active control ID
				activePopupControlID = popupControlID;
				
				// Display the dropdown
				DropdownMultiple
				(
					new Vector2(position.xMin, position.yMax),
					(values) =>
					{
						activeMultipleDropdownValues = values;
						activeDropdownChanged = true;
					},
					options != null ? options : getOptions(),
					selectedOptions,
					showNothingEverything
				);
			}

			if (popupControlID == activePopupControlID && activeDropdownChanged)
			{
				GUI.changed = true;
				activePopupControlID = -1;
				activeDropdownChanged = false;
				return activeMultipleDropdownValues;
			}
			else
			{
				return selectedOptions.ToHashSet();
			}
		}
		
		#endregion
	}
}
