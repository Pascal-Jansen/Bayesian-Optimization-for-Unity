// ColorWheelMarker.cs
// Renders an HSV color wheel into a RawImage and places a marker at the RGB's hue/saturation.
// Robust to edit-time and play-time timing. Uses 0..1 float RGB.

using System.Collections;
using UnityEngine;
using UnityEngine.UI;

[ExecuteAlways]
public class ColorWheelMarker : MonoBehaviour
{
    [Header("Wheel")]
    public RawImage wheelImage;                 // Assign a RawImage on a Canvas
    [Range(128, 2048)] public int wheelSize = 320;

    [Header("Marker")]
    public RectTransform marker;                // Assign a small Image/RectTransform as the marker
    public float markerRadius = 7f;

    [Header("Optional Swatch + Readout")]
    public Image swatch;                        // Optional
    public Text readout;                        // Optional

    private Texture2D _wheelTex;
    private RectTransform _wheelRect;
    private Vector2 _center;
    private float _radius;
    private bool _wheelBuilt;

    // ---------------- lifecycle ----------------
    void Awake()
    {
        // Try cache early if already assigned in Inspector
        if (wheelImage != null) _wheelRect = wheelImage.rectTransform;
    }

    void OnEnable()
    {
        EnsureInitialized();
        // Recompute geometry after first layout pass
        if (isActiveAndEnabled) StartCoroutine(DelayedGeometryCache());
    }

    void Start()
    {
        // Extra safety at runtime if something assigned late
        EnsureInitialized();
    }

    void OnValidate()
    {
        // Rebuild if params changed in the Inspector
        if (!isActiveAndEnabled) return;
        EnsureInitialized();
        BuildWheelTextureIfNeeded(forceRebuild: true);
        UpdateGeometryCache();
    }

    void OnRectTransformDimensionsChange()
    {
        // Canvas/layout changed
        UpdateGeometryCache();
    }

    IEnumerator DelayedGeometryCache()
    {
        // Wait a frame so layout system sets sizes
        yield return null;
#if UNITY_EDITOR
        if (!Application.isPlaying) { Canvas.ForceUpdateCanvases(); }
#endif
        UpdateGeometryCache();
    }

    // ---------------- public API ----------------
    /// <summary>Set marker using float RGB in [0..1]. Safe to call anytime.</summary>
    public void SetColor01(float r, float g, float b)
    {
        EnsureInitialized();

        r = Mathf.Clamp01(r);
        g = Mathf.Clamp01(g);
        b = Mathf.Clamp01(b);

        var color = new Color(r, g, b, 1f);
        if (swatch != null) swatch.color = color;

        // Convert and place marker
        Color.RGBToHSV(color, out float h, out float s, out float v);
        PlaceMarker(h, s);

        if (readout != null)
            readout.text = $"RGB ({r:0.00}, {g:0.00}, {b:0.00})  |  HSV ({h * 360f:0.0}°, {s * 100f:0}%, {v * 100f:0}%)";
    }

    [ContextMenu("Example: SetColor01(0.125, 0.5, 0.88)")]
    private void Example() => SetColor01(0.125f, 0.5f, 0.88f);

    // ---------------- internals ----------------
    private void EnsureInitialized()
    {
        if (wheelImage == null) return;

        if (_wheelRect == null)
            _wheelRect = wheelImage.rectTransform;

        BuildWheelTextureIfNeeded();
        if (_wheelRect != null && (_radius <= 0f || _center == Vector2.zero))
            UpdateGeometryCache();
    }

    private void BuildWheelTextureIfNeeded(bool forceRebuild = false)
    {
        if (wheelImage == null) return;

        bool sizeMismatch = _wheelTex == null || _wheelTex.width != wheelSize || _wheelTex.height != wheelSize;
        if (!_wheelBuilt || forceRebuild || sizeMismatch)
        {
            BuildWheelTexture();
            _wheelBuilt = true;
        }
    }

    private void BuildWheelTexture()
    {
        // Create/resize texture
        if (_wheelTex == null || _wheelTex.width != wheelSize || _wheelTex.height != wheelSize)
        {
            if (_wheelTex != null)
            {
#if UNITY_EDITOR
                if (!Application.isPlaying) DestroyImmediate(_wheelTex);
                else Destroy(_wheelTex);
#else
                Destroy(_wheelTex);
#endif
            }
            _wheelTex = new Texture2D(wheelSize, wheelSize, TextureFormat.RGBA32, false, true)
            {
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };
            wheelImage.texture = _wheelTex;
        }

        int W = wheelSize, H = wheelSize;
        float cx = (W - 1) * 0.5f;
        float cy = (H - 1) * 0.5f;
        float R = Mathf.Min(W, H) * 0.5f - 1f;

        var pixels = new Color32[W * H];
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                float dx = x - cx;
                float dy = y - cy;
                float rr = Mathf.Sqrt(dx * dx + dy * dy);
                int idx = y * W + x;

                if (rr <= R)
                {
                    float angle = Mathf.Atan2(dy, dx);           // [-π, π]
                    if (angle < 0f) angle += Mathf.PI * 2f;      // [0, 2π)
                    float h = angle / (Mathf.PI * 2f);           // [0,1)
                    float s = Mathf.Clamp01(rr / R);             // [0,1]
                    Color rgb = Color.HSVToRGB(h, s, 1f);        // V=1
                    pixels[idx] = (Color32)rgb;
                }
                else
                {
                    pixels[idx] = new Color32(0, 0, 0, 0);       // transparent outside
                }
            }
        }
        _wheelTex.SetPixels32(pixels);
        _wheelTex.Apply(false, false);
    }

    private void UpdateGeometryCache()
    {
        if (_wheelRect == null) return;
        var rect = _wheelRect.rect;

        // Guard zero-size while layout not ready; retry later
        if (rect.width <= 1f || rect.height <= 1f)
            return;

        _center = rect.center;                                   // local center
        _radius = Mathf.Min(rect.width, rect.height) * 0.5f - 1f;
    }

    private void PlaceMarker(float hue01, float sat01)
    {
        if (_wheelRect == null || marker == null) return;

        // Ensure marker is under the wheel for local anchoring
        if (marker.parent != _wheelRect)
            marker.SetParent(_wheelRect, worldPositionStays: false);

        if (_radius <= 0f)
        {
            // Try recompute if geometry not ready yet
            UpdateGeometryCache();
            if (_radius <= 0f) return;
        }

        float angle = hue01 * Mathf.PI * 2f;
        float rad = Mathf.Clamp01(sat01) * _radius;
        Vector2 localPos = _center + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * rad;

        marker.anchoredPosition = localPos;
    }
}
