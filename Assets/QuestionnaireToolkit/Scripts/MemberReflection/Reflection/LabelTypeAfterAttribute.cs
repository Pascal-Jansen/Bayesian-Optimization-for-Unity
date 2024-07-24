using System;

namespace QuestionnaireToolkit.Scripts.MemberReflection.Reflection
{
	/// <summary>
	/// Displays the type of the member after its name.
	/// </summary>
	[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property | AttributeTargets.Method, Inherited = true, AllowMultiple = false)]
	public sealed class LabelTypeAfterAttribute : Attribute
	{
		public LabelTypeAfterAttribute() { }
	}
}