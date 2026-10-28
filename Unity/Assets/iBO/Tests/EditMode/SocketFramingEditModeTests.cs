using System;
using System.Reflection;
using System.Text;
using BayesianOptimization.Scripts;
using NUnit.Framework;
using UnityEngine;

namespace BayesianOptimization.Tests.EditMode
{
    public class SocketFramingEditModeTests
    {
        private GameObject _object;
        private SocketNetwork _socket;

        [SetUp]
        public void SetUp()
        {
            _object = new GameObject("SocketFramingTest");
            _socket = _object.AddComponent<SocketNetwork>();
            SetField("_utf8Decoder", new UTF8Encoding(false).GetDecoder());
        }

        [TearDown]
        public void TearDown() => UnityEngine.Object.DestroyImmediate(_object);

        [Test]
        public void AcceptsSeveralFramesWithinTheirIndividualLimits()
        {
            SetField("_maxIncomingMessageBytes", 4L);
            Append(Encoding.UTF8.GetBytes("abc\nxyz\n"));
            Assert.That(BufferText(), Is.EqualTo("abc\nxyz\n"));
        }

        [Test]
        public void RejectsAnUnterminatedFrameBeforeAppendingBeyondTheLimit()
        {
            SetField("_maxIncomingMessageBytes", 4L);
            Append(Encoding.UTF8.GetBytes("abcd"));
            var error = Assert.Throws<TargetInvocationException>(() => Append(new byte[] { (byte)'e' }));
            Assert.That(error.InnerException, Is.TypeOf<InvalidOperationException>());
            Assert.That(BufferText(), Is.EqualTo("abcd"));
        }

        [Test]
        public void PreservesUtf8CodePointsSplitAcrossReads()
        {
            SetField("_maxIncomingMessageBytes", 5L);
            byte[] bytes = Encoding.UTF8.GetBytes("éé\n");
            Append(new[] { bytes[0] });
            Append(new[] { bytes[1], bytes[2], bytes[3], bytes[4] });
            Assert.That(BufferText(), Is.EqualTo("éé\n"));
        }

        [Test]
        public void CountsUtf8BytesAndTheNewlineDelimiter()
        {
            SetField("_maxIncomingMessageBytes", 4L);
            Append(Encoding.UTF8.GetBytes("éé"));
            var error = Assert.Throws<TargetInvocationException>(() => Append(new byte[] { (byte)'\n' }));
            Assert.That(error.InnerException, Is.TypeOf<InvalidOperationException>());
            Assert.That(BufferText(), Is.EqualTo("éé"));
        }

        private void SetField(string name, object value) =>
            typeof(SocketNetwork).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic).SetValue(_socket, value);

        private void Append(byte[] bytes) =>
            typeof(SocketNetwork).GetMethod("AppendReceivedBytes", BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(_socket, new object[] { bytes, bytes.Length });

        private string BufferText() =>
            typeof(SocketNetwork).GetField("_lineBuf", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(_socket).ToString();
    }
}
