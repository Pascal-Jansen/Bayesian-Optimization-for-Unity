#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace QuestionnaireToolkit.Scripts
{
    public static class TagsAndLayers
    {
        private static int maxTags = 10000;
        private static int maxLayers = 31;

        private static string[] qtTags =
        {
            "QTSelectionHighlight",
            "QTSlider",
            "QTQuestionPage",
            "QTQuestionnaireManager",
            "QTMultipleChoice",
            "QTOptionOther",
            "QTButton",
            "QTCheckboxes",
            "QTCheckboxesGrid",
            "QTGridOption",
            "QTDropdown",
            "QTGridRowHeader",
            "QTImage",
            "QTLinearScale",
            "QTMultipleChoiceGrid",
            "QTManager",
            "QTText",
            "QTTextInput",
            "QTVideo"
        };
        
        /// <summary>
        /// Adds the tag.
        /// </summary>
        /// <returns><c>true</c>, if tag was added, <c>false</c> otherwise.</returns>
        /// <param name="tagName">Tag name.</param>
        public static bool CreateTag(string tagName)
        {
            // Open tag manager
            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            // Tags Property
            SerializedProperty tagsProp = tagManager.FindProperty("tags");
            if (tagsProp.arraySize >= maxTags)
            {
                Debug.Log("No more tags can be added to the Tags property. You have " + tagsProp.arraySize + " tags");
                return false;
            }
            // if not found, add it
            if (!PropertyExists(tagsProp, 0, tagsProp.arraySize, tagName))
            {
                int index = tagsProp.arraySize;
                // Insert new array element
                tagsProp.InsertArrayElementAtIndex(index);
                SerializedProperty sp = tagsProp.GetArrayElementAtIndex(index);
                // Set array element to tagName
                sp.stringValue = tagName;
                //Debug.Log("Tag: " + tagName + " has been added");
                // Save settings
                tagManager.ApplyModifiedProperties();
 
                return true;
            }
            else
            {
                //Debug.Log ("Tag: " + tagName + " already exists");
            }
            return false;
        }
 
        public static string NewTag(string name)
        {
            CreateTag(name);
 
            if (string.IsNullOrEmpty(name))
            {
                name = "Untagged";
            }
 
            return name;
        }
 
        /// <summary>
        /// Removes the tag.
        /// </summary>
        /// <returns><c>true</c>, if tag was removed, <c>false</c> otherwise.</returns>
        /// <param name="tagName">Tag name.</param>
        ///
        public static bool RemoveTag(string tagName)
        {
 
            // Open tag manager
            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
 
            // Tags Property
            SerializedProperty tagsProp = tagManager.FindProperty("tags");
 
            if (PropertyExists(tagsProp, 0, tagsProp.arraySize, tagName))
            {
                SerializedProperty sp;
 
                for (int i = 0, j = tagsProp.arraySize; i < j; i++)
                {
 
                    sp = tagsProp.GetArrayElementAtIndex(i);
                    if (sp.stringValue == tagName)
                    {
                        tagsProp.DeleteArrayElementAtIndex(i);
                        Debug.Log("Tag: " + tagName + " has been removed");
                        // Save settings
                        tagManager.ApplyModifiedProperties();
                        return true;
                    }
 
                }
            }
 
            return false;
 
        }

        public static void RefreshQtTags()
        {
            if (EditorApplication.isCompiling || EditorApplication.isPlayingOrWillChangePlaymode) return;
            
            // Open tag manager
            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            // Tags Property
            SerializedProperty tagsProp = tagManager.FindProperty("tags");
            if (tagsProp.arraySize >= maxTags)
            {
                Debug.Log("No more tags can be added to the Tags property. You have " + tagsProp.arraySize + " tags");
                return;
            }
            foreach (var tagName in qtTags)
            {
                if (!PropertyExists(tagsProp, 0, tagsProp.arraySize, tagName))
                {
                    int index = tagsProp.arraySize;
                    tagsProp.InsertArrayElementAtIndex(index);
                    SerializedProperty sp = tagsProp.GetArrayElementAtIndex(index);
                    sp.stringValue = tagName;
                    tagManager.ApplyModifiedProperties();
                }
            }
        }
        /// <summary>
        /// Checks to see if tag exists.
        /// </summary>
        /// <returns><c>true</c>, if tag exists, <c>false</c> otherwise.</returns>
        /// <param name="tagName">Tag name.</param>
        public static bool TagExists(string tagName)
        {
            // Open tag manager
            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
 
            // Layers Property
            SerializedProperty tagsProp = tagManager.FindProperty("tags");
            return PropertyExists(tagsProp, 0, maxTags, tagName);
        }
        /// <summary>
        /// Adds the layer.
        /// </summary>
        /// <returns><c>true</c>, if layer was added, <c>false</c> otherwise.</returns>
        /// <param name="layerName">Layer name.</param>
        public static bool CreateLayer(string layerName)
        {
            // Open tag manager
            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            // Layers Property
            SerializedProperty layersProp = tagManager.FindProperty("layers");
            if (!PropertyExists(layersProp, 0, maxLayers, layerName))
            {
                SerializedProperty sp;
                // Start at layer 9th index -> 8 (zero based) => first 8 reserved for unity / greyed out
                for (int i = 8, j = maxLayers; i < j; i++)
                {
                    sp = layersProp.GetArrayElementAtIndex(i);
                    if (sp.stringValue == "")
                    {
                        // Assign string value to layer
                        sp.stringValue = layerName;
                        Debug.Log("Layer: " + layerName + " has been added");
                        // Save settings
                        tagManager.ApplyModifiedProperties();
                        return true;
                    }
                    if (i == j)
                        Debug.Log("All allowed layers have been filled");
                }
            }
            else
            {
                //Debug.Log ("Layer: " + layerName + " already exists");
            }
            return false;
        }
 
        public static string NewLayer(string name)
        {
            if (!string.IsNullOrEmpty(name))
            {
                CreateLayer(name);
            }
 
            return name;
        }
 
        /// <summary>
        /// Removes the layer.
        /// </summary>
        /// <returns><c>true</c>, if layer was removed, <c>false</c> otherwise.</returns>
        /// <param name="layerName">Layer name.</param>
        public static bool RemoveLayer(string layerName)
        {
 
            // Open tag manager
            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
 
            // Tags Property
            SerializedProperty layersProp = tagManager.FindProperty("layers");
 
            if (PropertyExists(layersProp, 0, layersProp.arraySize, layerName))
            {
                SerializedProperty sp;
 
                for (int i = 0, j = layersProp.arraySize; i < j; i++)
                {
 
                    sp = layersProp.GetArrayElementAtIndex(i);
 
                    if (sp.stringValue == layerName)
                    {
                        sp.stringValue = "";
                        Debug.Log("Layer: " + layerName + " has been removed");
                        // Save settings
                        tagManager.ApplyModifiedProperties();
                        return true;
                    }
 
                }
            }
 
            return false;
 
        }
        /// <summary>
        /// Checks to see if layer exists.
        /// </summary>
        /// <returns><c>true</c>, if layer exists, <c>false</c> otherwise.</returns>
        /// <param name="layerName">Layer name.</param>
        public static bool LayerExists(string layerName)
        {
            // Open tag manager
            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
 
            // Layers Property
            SerializedProperty layersProp = tagManager.FindProperty("layers");
            return PropertyExists(layersProp, 0, maxLayers, layerName);
        }
        /// <summary>
        /// Checks if the value exists in the property.
        /// </summary>
        /// <returns><c>true</c>, if exists was propertyed, <c>false</c> otherwise.</returns>
        /// <param name="property">Property.</param>
        /// <param name="start">Start.</param>
        /// <param name="end">End.</param>
        /// <param name="value">Value.</param>
        private static bool PropertyExists(SerializedProperty property, int start, int end, string value)
        {
            if (property.arraySize == 0)
            {
                return false;
            }

            for (int i = start; i < end; i++)
            {
                SerializedProperty t = property.GetArrayElementAtIndex(i);
                if (t.stringValue.Equals(value))
                {
                    return true;
                }
            }
            return false;
        }
    }
}
#endif