using System.IO;
using NUnit.Framework;
using QuestionnaireToolkit.Scripts;

namespace BOforUnity.Tests.EditMode
{
    public class LogDataFolderUtilityEditModeTests
    {
        private string _root;

        [SetUp]
        public void SetUp()
        {
            // A fresh root per test: reservations are process-wide and keyed by root.
            _root = Path.Combine(Path.GetTempPath(), "bo4unity_logroot_" + Path.GetRandomFileName());
            Directory.CreateDirectory(_root);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_root))
                Directory.Delete(_root, recursive: true);
        }

        [Test]
        public void ReusesTheUserFolderForAnotherCondition()
        {
            // An earlier condition of the same participant already wrote P01/static.
            Directory.CreateDirectory(Path.Combine(_root, "P01", "static"));

            string token = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(
                _root, "P01", "HITL MOBO", allowExistingRequestedUserFolder: true);

            Assert.That(token, Is.EqualTo("P01"));
            Assert.That(Directory.Exists(Path.Combine(_root, "P01", "HITL MOBO")), Is.True);
        }

        [Test]
        public void AddsASuffixWhenTheSameConditionFolderExists()
        {
            Directory.CreateDirectory(Path.Combine(_root, "P02", "static"));

            string token = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(
                _root, "P02", "static", allowExistingRequestedUserFolder: true, allowExistingConditionFolder: false);

            Assert.That(token, Is.EqualTo("P02_1"));
            Assert.That(Directory.Exists(Path.Combine(_root, "P02_1", "static")), Is.True);
        }

        [Test]
        public void WithoutPermissionAnExistingUserFolderGetsASuffix()
        {
            Directory.CreateDirectory(Path.Combine(_root, "P05", "static"));

            string token = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(_root, "P05", "random");

            Assert.That(token, Is.EqualTo("P05_1"));
        }

        [Test]
        public void RootsWithAndWithoutTrailingSeparatorShareOneReservation()
        {
            string withSeparator = _root + Path.DirectorySeparatorChar;

            string first = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(withSeparator, "P03", "c");
            // The folder P03/c exists now; only a shared reservation returns the same token.
            string second = LogDataFolderUtility.GetOrCreateUserFolderTokenForCondition(_root, "P03", "c");

            Assert.That(first, Is.EqualTo("P03"));
            Assert.That(second, Is.EqualTo("P03"), "The trailing separator must not create a second reservation.");
            Assert.That(Directory.Exists(Path.Combine(_root, "P03_1")), Is.False);
        }

        [Test]
        public void NormalizeRootRemovesTrailingSeparatorsOnly()
        {
            string normalized = LogDataFolderUtility.NormalizeRoot(_root + Path.DirectorySeparatorChar);

            Assert.That(normalized, Is.EqualTo(Path.GetFullPath(_root).TrimEnd(Path.DirectorySeparatorChar)));
            Assert.That(LogDataFolderUtility.IsSameRoot(_root, _root + Path.AltDirectorySeparatorChar), Is.True);

            string fileSystemRoot = Path.GetPathRoot(Path.GetFullPath(_root));
            Assert.That(LogDataFolderUtility.NormalizeRoot(fileSystemRoot), Is.EqualTo(fileSystemRoot),
                "A bare file-system root keeps its separator.");
        }

        [Test]
        public void NormalizesFolderTokens()
        {
            Assert.That(LogDataFolderUtility.NormalizeLogFolderToken("a/b:c"), Is.EqualTo("a_b_c"));
            Assert.That(LogDataFolderUtility.NormalizeLogFolderToken("P\u0080Q\u009F1"), Is.EqualTo("P_Q_1"),
                "C1 control characters (0x7F-0x9F) are replaced like C0 ones.");
            Assert.That(LogDataFolderUtility.NormalizeLogFolderToken("  "), Is.EqualTo("-1"));
            Assert.That(LogDataFolderUtility.NormalizeLogFolderToken(".."), Is.EqualTo("-1"));
            Assert.That(LogDataFolderUtility.NormalizeLogFolderToken("007"), Is.EqualTo("007"));
        }
    }
}
