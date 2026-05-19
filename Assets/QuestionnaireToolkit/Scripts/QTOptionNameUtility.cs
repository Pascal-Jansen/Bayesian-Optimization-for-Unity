using System.Globalization;

namespace QuestionnaireToolkit.Scripts
{
    internal static class QTOptionNameUtility
    {
        public static string GetValue(string optionName)
        {
            if (string.IsNullOrEmpty(optionName))
                return string.Empty;

            int separatorIndex = optionName.IndexOf('_');
            return separatorIndex < 0 ? optionName : optionName.Substring(0, separatorIndex);
        }

        public static string GetText(string optionName)
        {
            if (string.IsNullOrEmpty(optionName))
                return string.Empty;

            int separatorIndex = optionName.IndexOf('_');
            if (separatorIndex < 0 || separatorIndex >= optionName.Length - 1)
                return string.Empty;

            return optionName.Substring(separatorIndex + 1);
        }

        public static string Compose(string value, string text)
        {
            return (value ?? string.Empty) + "_" + (text ?? string.Empty);
        }

        public static string RenameValue(string optionName, int value)
        {
            return Compose(value.ToString(CultureInfo.InvariantCulture), GetText(optionName));
        }
    }
}
