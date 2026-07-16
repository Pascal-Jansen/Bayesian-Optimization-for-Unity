using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using BOforUnity;
using BOforUnity.Scripts;

namespace BOforUnity.Tests.EditMode
{
    public class OptimizerKeyMatchingEditModeTests
    {
        private GameObject _go;
        private BoForUnityManager _manager;
        private Optimizer _optimizer;

        [SetUp]
        public void SetUp()
        {
            _go = new GameObject("OptimizerKeyMatchingTest");
            _manager = _go.AddComponent<BoForUnityManager>();
            _optimizer = _go.AddComponent<Optimizer>();
            _optimizer.Start(); // resolves the manager reference (no play mode in EditMode tests)

            _manager.objectives = new List<ObjectiveEntry>
            {
                new ObjectiveEntry("usability", new ObjectiveArgs(0f, 10f, false, 1)),
                new ObjectiveEntry("usability1", new ObjectiveArgs(0f, 10f, false, 1)),
                new ObjectiveEntry("trust", new ObjectiveArgs(0f, 10f, false, 1))
            };
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_go);
        }

        [Test]
        public void MatchesExactAndBoundaryDelimitedKeys()
        {
            Assert.That(_optimizer.HasObjectiveMatch("usability"), Is.True);
            Assert.That(_optimizer.HasObjectiveMatch("Q1_trust"), Is.True, "Underscore boundary should match.");
            Assert.That(_optimizer.HasObjectiveMatch("trustScore"), Is.True, "camelCase boundary should match.");
        }

        [Test]
        public void DoesNotMatchKeysEmbeddedInsideWords()
        {
            Assert.That(_optimizer.HasObjectiveMatch("reusability"), Is.False,
                "A key embedded inside a longer word must not match.");
            Assert.That(_optimizer.HasObjectiveMatch("distrustful"), Is.False);
            Assert.That(_optimizer.HasObjectiveMatch("unrelated"), Is.False);
        }

        [Test]
        public void AddObjectiveValuePrefersMostSpecificKey()
        {
            _optimizer.AddObjectiveValue("usability1", 4.0f);

            ObjectiveArgs general = _optimizer.GetObjective("usability");
            ObjectiveArgs specific = _optimizer.GetObjective("usability1");

            Assert.That(specific.values, Has.Count.EqualTo(1),
                "The value must land on the most specific (longest) matching key.");
            Assert.That(specific.values[0], Is.EqualTo(4.0f).Within(1e-6f));
            Assert.That(general.values, Is.Empty,
                "The shorter key must not receive the value.");
        }
    }
}
