using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using NUnit.Framework;
using UnityEngine;

namespace DboTorch.Tests
{
    public class DboClientEditModeTests
    {
        [Test]
        public void DecodesUnicodeAndControlCharacterEscapesInErrors()
        {
            object reply = Parse(@"{""ok"":false,""error"":""\u00fc \ud83d\ude42 \n \""""}", "Acknowledgment");
            Assert.That(Field<string>(reply, "Error"), Is.EqualTo("ü 🙂 \n \""));
        }

        [Test]
        public void PreservesEveryNumericCoordinateAndSuggestionMetadata()
        {
            object reply = Parse(@"{""ok"":true,""x"":[1.25e-3,-2,3],""iteration"":7,""is_validation"":true}", "Suggestion");
            var suggestion = Field<DboSuggestion>(reply, "Suggestion");
            Assert.That(suggestion.X, Is.EqualTo(new[] { 0.00125f, -2f, 3f }));
            Assert.That(suggestion.Iteration, Is.EqualTo(7));
            Assert.That(suggestion.IsValidation, Is.True);
        }

        [TestCase("null")]
        [TestCase("\"bad\"")]
        [TestCase("true")]
        [TestCase("{}")]
        [TestCase("[]")]
        [TestCase("NaN")]
        [TestCase("Infinity")]
        [TestCase("1e100")]
        public void RejectsInvalidCoordinatesWithoutShorteningTheDesign(string coordinate)
        {
            Invalid("{\"ok\":true,\"x\":[1," + coordinate + ",3],\"iteration\":1,\"is_validation\":false}", "Suggestion");
        }

        [TestCase(@"{""ok"":true,""x"":[],""iteration"":1,""is_validation"":false}")]
        [TestCase(@"{""ok"":true,""x"":[1],""is_validation"":false}")]
        [TestCase(@"{""ok"":true,""x"":[1],""iteration"":1.5,""is_validation"":false}")]
        [TestCase(@"{""ok"":true,""x"":[1],""iteration"":1,""is_validation"":""false""}")]
        public void RejectsMalformedSuggestionFields(string json) => Invalid(json, "Suggestion");

        [Test]
        public void ParsesFinitePredictionValues()
        {
            object reply = Parse(@"{""ok"":true,""mean"":-0.5,""std"":0.25}", "Prediction");
            var prediction = Field<DboPrediction>(reply, "Prediction");
            Assert.That(prediction.Mean, Is.EqualTo(-0.5f));
            Assert.That(prediction.StdDev, Is.EqualTo(0.25f));
            Invalid(@"{""ok"":true,""mean"":0,""std"":-1}", "Prediction");
        }

        [TestCase(@"{""ok"":""true""}")]
        [TestCase(@"{""ok"":true,""ok"":false}")]
        public void RejectsInvalidReplyEnvelope(string json) => Invalid(json, "Acknowledgment");

        [Test]
        public void MalformedArrayReachesErrorCallbackWithoutReportingSuccess()
        {
            var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            var go = new GameObject("StandaloneDboClientParserTest");
            var client = go.AddComponent<DboClient>();
            client.Port = ((IPEndPoint)listener.LocalEndpoint).Port;
            client.TimeoutSeconds = 2f;
            string error = null;
            bool succeeded = false;
            Exception serverError = null;
            var server = new Thread(() =>
            {
                try
                {
                    using (var peer = listener.AcceptTcpClient())
                    using (var stream = peer.GetStream())
                    using (var reader = new StreamReader(stream, new UTF8Encoding(false), false, 1024, true))
                    using (var writer = new StreamWriter(stream, new UTF8Encoding(false), 1024, true) { AutoFlush = true })
                    {
                        reader.ReadLine();
                        writer.WriteLine(@"{""ok"":true,""x"":[1,""bad"",3],""iteration"":1,""is_validation"":false}");
                    }
                }
                catch (Exception exception) { serverError = exception; }
            }) { IsBackground = true };

            try
            {
                server.Start();
                client.Suggest(_ => succeeded = true, message => error = message);
                var timer = Stopwatch.StartNew();
                var update = typeof(DboClient).GetMethod("Update", BindingFlags.Instance | BindingFlags.NonPublic);
                while (error == null && !succeeded && timer.Elapsed.TotalSeconds < 5)
                {
                    update.Invoke(client, null);
                    Thread.Sleep(10);
                }
                Assert.That(error, Does.Contain("x[1]"));
                Assert.That(succeeded, Is.False);
                Assert.That(serverError, Is.Null);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
                listener.Stop();
                server.Join(1000);
            }
        }

        private static object Parse(string json, string kind)
        {
            var replyKind = typeof(DboClient).GetNestedType("ReplyKind", BindingFlags.NonPublic);
            return typeof(DboClient).GetMethod("ParseReply", BindingFlags.Static | BindingFlags.NonPublic)
                .Invoke(null, new[] { (object)json, Enum.Parse(replyKind, kind) });
        }

        private static void Invalid(string json, string kind)
        {
            var error = Assert.Throws<TargetInvocationException>(() => Parse(json, kind));
            Assert.That(error.InnerException, Is.Not.Null);
        }

        private static T Field<T>(object value, string name) =>
            (T)value.GetType().GetField(name).GetValue(value);
    }
}
