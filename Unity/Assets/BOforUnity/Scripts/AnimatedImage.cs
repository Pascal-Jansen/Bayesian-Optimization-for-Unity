using UnityEngine;
using UnityEngine.UI;

namespace BOforUnity.Scripts
{
    public class AnimatedImage : MonoBehaviour
    {
        public Sprite[] frames; // Array to hold the frames of the animation
        public float frameRate = 0.1f; // Time between frames

        private Image image;
        private int currentFrame;
        private float timer;

        void Start()
        {
            image = GetComponent<Image>();
            if (frames.Length > 0)
            {
                image.sprite = frames[0];
            }
        }

        void Update()
        {
            if (frames.Length > 0)
            {
                timer += Time.deltaTime;
                if (timer >= frameRate)
                {
                    timer -= frameRate;
                    currentFrame = (currentFrame + 1) % frames.Length;
                    image.sprite = frames[currentFrame];
                }
            }
        }
    }
}