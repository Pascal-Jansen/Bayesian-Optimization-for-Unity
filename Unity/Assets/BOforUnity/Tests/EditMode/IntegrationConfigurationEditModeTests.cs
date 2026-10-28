using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using BayesianOptimization.Examples;
using BayesianOptimization.Scripts;
using NUnit.Framework;
using UnityEngine;

namespace BayesianOptimization.Tests.EditMode
{
    public class IntegrationConfigurationEditModeTests
    {
        [TestCase(0)]
        [TestCase(-1)]
        [TestCase(65536)]
        public void RefusesInvalidLoopbackPorts(int port)
        {
            var target = new GameObject("PortValidationTest");
            try
            {
                var manager = target.AddComponent<BoForUnityManager>();
                manager.port = port;
                Assert.That(string.Join(" ", BoConfigValidator.GetErrors(manager)), Does.Contain("TCP port"));
            }
            finally { UnityEngine.Object.DestroyImmediate(target); }
        }

        [TestCase(1)]
        [TestCase(56001)]
        [TestCase(65535)]
        public void AcceptsValidLoopbackPorts(int port)
        {
            var target = new GameObject("PortValidationTest");
            try
            {
                var manager = target.AddComponent<BoForUnityManager>();
                manager.port = port;
                Assert.That(string.Join(" ", BoConfigValidator.GetErrors(manager)), Does.Not.Contain("TCP port"));
            }
            finally { UnityEngine.Object.DestroyImmediate(target); }
        }

        [Test]
        public void PassesTheConfiguredPortToPython()
        {
            var start = new ProcessStartInfo();
            typeof(PythonStarter).GetMethod("ConfigurePythonRuntimeEnvironment", BindingFlags.Static | BindingFlags.NonPublic)
                .Invoke(null, new object[] { start, "test-log-root", false, "botorch", 56002 });
            Assert.That(start.Environment["BO_PORT"], Is.EqualTo("56002"));
        }

        [Test]
        public void ProtocolDtoStillSerializesDictionaryAndNullableIteration()
        {
            Type dtoType = typeof(SocketNetwork).Assembly.GetType("BayesianOptimization.Scripts.ParametersMsg");
            object dto = Activator.CreateInstance(dtoType);
            dtoType.GetField("type").SetValue(dto, "parameters");
            dtoType.GetField("values").SetValue(dto, new Dictionary<string, float> { { "x", 0.25f } });
            dtoType.GetField("iteration").SetValue(dto, (int?)7);
            string json = (string)typeof(SocketNetwork).GetMethod("SerializeJson", BindingFlags.Static | BindingFlags.NonPublic)
                .Invoke(null, new[] { dto });
            Assert.That(json, Does.Contain("\"values\":{\"x\":0.25}"));
            Assert.That(json, Does.Contain("\"iteration\":7"));
        }

        [Test]
        public void AnalyticObjectiveHasTheDocumentedMinimumAndFiniteRange()
        {
            Assert.That(AnalyticOptimizationExample.Evaluate(0.25f, 0.75f), Is.Zero);
            Assert.That(AnalyticOptimizationExample.Evaluate(1f, 0f), Is.EqualTo(1.125f));
        }
    }
}
