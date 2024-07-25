using UnityEngine;

namespace BOforUnity.Examples
{
    public class ObjectVariableExposer : MonoBehaviour
    {
        // Transform variables
        public float positionX;
        public float positionY;
        public float positionZ;
        public float rotationX;
        public float rotationY;
        public float rotationZ;
        public float scaleX;
        public float scaleY;
        public float scaleZ;

        // Renderer variables
        public float colorR;
        public float colorG;
        public float colorB;
        public float colorA;

        // GameObject active state
        public bool isActive;

        private MeshRenderer _renderer;
        
        void Start()
        {
            _renderer = GetComponent<MeshRenderer>();

            /*
            // Initialize Transform variables
            positionX = transform.position.x;
            positionY = transform.position.y;
            positionZ = transform.position.z;
            rotationX = transform.eulerAngles.x;
            rotationY = transform.eulerAngles.y;
            rotationZ = transform.eulerAngles.z;
            scaleX = transform.localScale.x;
            scaleY = transform.localScale.y;
            scaleZ = transform.localScale.z;

            // Initialize Renderer variables
            if (_renderer != null)
            {
                colorR = _renderer.material.color.r;
                colorG = _renderer.material.color.g;
                colorB = _renderer.material.color.b;
                colorA = _renderer.material.color.a;
            }

            // Initialize GameObject active state
            isActive = gameObject.activeSelf;
            */
            
            GameObject.FindWithTag("BOforUnityManager").GetComponent<BoForUnityManager>().optimizer.UpdateDesignParameters();
        }
        
        // This Update is only needed to listen on any changes to these parameters made by the optimizer
        void Update()
        {
            transform.position = new Vector3(positionX,positionY,positionZ);

            transform.eulerAngles = new Vector3(rotationX, rotationY, rotationZ);

            transform.localScale = new Vector3(scaleX,scaleY,scaleZ);

            _renderer.material.color = new Color(colorR, colorG, colorB, colorA);

            _renderer.enabled = isActive;
        }
    }
}