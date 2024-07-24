using System.Collections.Generic;
using System.Linq;
using UnityEditor;

namespace Ludiq.Controls.Editor
{
	public static class Extensions
	{
		/// <summary>
		/// Splits a given property into each of its multiple values.
		/// If it has a single value, only the same property is returned.
		/// </summary>
		public static IEnumerable<SerializedProperty> Multiple(this SerializedProperty property)
		{
			if (property.hasMultipleDifferentValues)
			{
				return property.serializedObject.targetObjects.Select(o => new SerializedObject(o).FindProperty(property.propertyPath));
			}
			else
			{
				return new[] { property };
			}
		}

		public static HashSet<T> ToHashSet<T>(this IEnumerable<T> enumerable)
		{
			return new HashSet<T>(enumerable);
		}

		public static ICollection<T> CacheToCollection<T>(this IEnumerable<T> enumerable)
		{
			if (enumerable is ICollection<T>)
			{
				return (ICollection<T>)enumerable;
			}
			else
			{
				return enumerable.ToList();
			}
		}
	}
}