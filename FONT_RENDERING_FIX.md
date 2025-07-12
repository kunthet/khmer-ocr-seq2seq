# Font Rendering Issue Fix

## Issue Description
The generated synthetic images were not rendering Khmer text properly due to a font matching bug in the `TextRenderer.render_text_pil()` method.

## Root Cause
When font paths like `fonts/KhmerOSmuollight.ttf` were passed to the renderer, the font matching logic failed to properly extract the font name and match it with the loaded PIL fonts. This caused all text to render with the first available font (usually `KhmerOS`) instead of the specified font.

### Original Problematic Code
```python
# In render_text_pil method
if font_name:
    # Try to match font by name or path
    for name, f in self.pil_fonts:
        if name == font_name or font_name in name:
            font = f
            break
```

The issue was that:
- `font_name` was `"fonts/KhmerOSmuollight.ttf"`
- `name` was `"KhmerOSmuollight"`
- Neither `name == font_name` nor `font_name in name` was true
- So it defaulted to the first available font

## Fix Applied
Updated the font matching logic to properly extract font names from paths:

```python
# In render_text_pil method
if font_name:
    # Extract font name from path if it's a path
    if os.path.exists(font_name):
        # It's a file path, extract the font name
        base_name = os.path.basename(font_name).replace('.ttf', '').replace('.otf', '')
    else:
        # It's already a font name
        base_name = font_name
    
    # Try to match font by name
    for name, f in self.pil_fonts:
        if name == base_name:
            font = f
            break
```

## Results
- Font paths like `fonts/KhmerOSmuollight.ttf` now correctly extract to `KhmerOSmuollight`
- Exact font name matching ensures the correct font is selected
- All generated images now render with the intended Khmer fonts
- Text rendering quality is significantly improved

## Files Modified
- `src/data/text_renderer.py`: Fixed font matching logic in `render_text_pil()` method
- `docs/changes.md`: Updated documentation with fix details

## Verification
- Regenerated all synthetic images (150 samples) with the fix
- All tests pass successfully
- Font matching now works correctly for all font paths
- Khmer text renders properly with the specified fonts

## Impact
This fix ensures that the synthetic image generation system produces high-quality training data with proper font diversity, which is crucial for training a robust Khmer OCR model. 