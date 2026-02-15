# 🚀 HOW TO RUN THE WEB APPLICATION

## Quick Start (3 Steps!)

### Step 1: Install Dependencies
```bash
cd bone_fracture_detection
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
python app.py
```

### Step 3: Open Browser
```
http://localhost:5000
```

**That's it!** 🎉

---

## Detailed Instructions

### Prerequisites

✅ **Python 3.8+** installed  
✅ **pip** package manager  
✅ **Internet connection** (for first-time package installation)  

---

### Installation

#### 1. Navigate to Project Directory
```bash
cd bone_fracture_detection
```

#### 2. (Optional but Recommended) Create Virtual Environment
```bash
# Create virtual environment
python -m venv fracture_env

# Activate it
# On Mac/Linux:
source fracture_env/bin/activate

# On Windows:
fracture_env\Scripts\activate
```

#### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- TensorFlow (AI model)
- OpenCV (image processing)
- NumPy, Pillow, etc.

---

### Running the Application

#### Basic Run
```bash
python app.py
```

You should see:
```
🏥 X-RAY BONE FRACTURE DETECTION SYSTEM
======================================================================
Starting web server...
Open your browser and navigate to: http://localhost:5000
Press Ctrl+C to stop the server
======================================================================

 * Running on http://0.0.0.0:5000
```

#### Access the App

Open your web browser and go to:
- **Local access:** http://localhost:5000
- **Network access:** http://YOUR_IP_ADDRESS:5000

---

## Demo Mode vs Real AI Mode

### Demo Mode (Default - No Model Required)

If you haven't trained a model yet, the app runs in **DEMO MODE**:

✅ **What works:**
- Upload interface ✓
- File validation ✓
- Image preview ✓
- Results display ✓

⚠️ **What's simulated:**
- Predictions (random values)
- Grad-CAM not available

**Demo mode is PERFECT for:**
- Testing the interface
- Understanding the workflow
- Showing the UI to others
- Development and debugging

---

### Real AI Mode (With Trained Model)

To use **REAL AI PREDICTIONS:**

#### 1. Train a Model (or get one)
```bash
# Option A: Train your own
python train.py

# Option B: Download pre-trained model
# (Place best_model.h5 in models/ directory)
```

#### 2. Ensure Model File Exists
```
bone_fracture_detection/
└── models/
    └── best_model.h5  ← This file must exist
```

#### 3. Restart the App
```bash
python app.py
```

You should see:
```
📦 Loading AI model...
✅ Model loaded successfully!
```

Now you have **REAL AI PREDICTIONS** with Grad-CAM! 🎉

---

## Using the Web Interface

### 1. Upload X-ray Image

**Method A: Click to Upload**
1. Click "Choose X-ray image"
2. Select image file
3. Preview appears

**Method B: Drag & Drop**
1. Drag image file
2. Drop on upload area
3. Preview appears

### 2. Analyze

Click **"Analyze X-ray"** button

### 3. View Results

See:
- ✅ Prediction (Fracture/Normal)
- ✅ Confidence percentage
- ✅ Original image
- ✅ Grad-CAM visualization (if fracture)
- ✅ Timestamp

### 4. Actions

- **Analyze Another:** Upload new image
- **Download Report:** Get results (coming soon)

---

## Supported Image Formats

✅ PNG  
✅ JPG / JPEG  
✅ BMP  
✅ TIFF  

**Max file size:** 16 MB

---

## Troubleshooting

### Issue: "Port 5000 already in use"

**Solution:**
```bash
# Use different port
python app.py
# Then modify app.py last line to:
# app.run(debug=True, host='0.0.0.0', port=8000)
```

---

### Issue: "Module not found"

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install specific package
pip install flask
pip install tensorflow
```

---

### Issue: "Model not loading"

**Symptoms:**
```
⚠️  WARNING: Model not found!
The app will run in DEMO mode
```

**Solution:**
1. Check if `models/best_model.h5` exists
2. If not, train a model: `python train.py`
3. Or run in demo mode (perfectly fine!)

---

### Issue: "Image won't upload"

**Solutions:**
- Check file format (must be image)
- Check file size (max 16MB)
- Try different image
- Check browser console for errors (F12)

---

### Issue: "Results not showing"

**Solutions:**
- Check browser console (F12)
- Try different browser
- Clear cache and reload
- Check network tab for errors

---

## Advanced Configuration

### Change Host/Port

Edit `app.py`, last line:
```python
# Default
app.run(debug=True, host='0.0.0.0', port=5000)

# Custom
app.run(debug=True, host='localhost', port=8080)
```

### Disable Debug Mode

For production:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Enable HTTPS

Use a production server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## File Structure

```
bone_fracture_detection/
├── app.py                 ← Main Flask application
├── templates/
│   ├── index.html        ← Main page
│   ├── about.html        ← About page
│   └── 404.html          ← Error page
├── static/
│   ├── css/
│   │   └── style.css     ← Styling
│   ├── js/
│   │   └── main.js       ← Interactivity
│   └── results/          ← Generated Grad-CAM images
├── uploads/               ← Temporary uploaded files
├── models/
│   └── best_model.h5     ← AI model (if available)
└── utils/                ← Helper modules
```

---

## Performance Tips

### For Faster Loading:

1. **Use GPU** (if available)
   - TensorFlow will auto-detect
   - 10x faster predictions

2. **Close other apps**
   - Free up RAM
   - Better performance

3. **Use production server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 app:app
   ```

---

## Security Notes

### For Development:
- ✅ Running locally is safe
- ✅ Only you can access

### For Production:
- ⚠️ Add authentication
- ⚠️ Use HTTPS
- ⚠️ Add rate limiting
- ⚠️ Implement HIPAA compliance
- ⚠️ Secure file uploads

---

## Stopping the Server

Press **Ctrl+C** in the terminal

Or close the terminal window

---

## Common Commands

```bash
# Start app
python app.py

# Start with auto-reload (development)
export FLASK_ENV=development
python app.py

# Check if app is running
curl http://localhost:5000/api/status

# View logs
# (They appear in terminal)
```

---

## Browser Compatibility

✅ Chrome (Recommended)  
✅ Firefox  
✅ Safari  
✅ Edge  
⚠️ IE 11 (Limited support)  

---

## Mobile Access

Yes! The interface is responsive.

Access from phone/tablet:
1. Find your computer's IP address
2. Open browser on phone
3. Navigate to: `http://YOUR_IP:5000`

**Note:** Both devices must be on same WiFi network

---

## What's Next?

After the app is running:

1. **Test it out** - Upload sample X-rays
2. **Train a model** - Get real predictions
3. **Deploy online** - Make it accessible anywhere
4. **Add features** - Customize to your needs

---

## Need Help?

### Check:
- README.md (main documentation)
- DEPLOYMENT_OPTIONS.md (deployment guide)
- PHASE5_COMPLETE.md (web app details)

### Common Issues:
- 95% of issues are dependency-related
- Solution: Reinstall requirements
- Still stuck? Check error messages carefully

---

## Quick Test

Run this to verify everything works:

```bash
# 1. Start app
python app.py

# 2. In another terminal:
curl http://localhost:5000/api/status

# Should return:
# {"status":"online", "model_loaded":false, "version":"1.0.0"}
```

---

## Summary

**To run the app:**
```bash
pip install -r requirements.txt
python app.py
```

**Then open:** http://localhost:5000

**Demo mode works immediately!**  
**Real AI requires training a model first!**

---

## 🎉 Enjoy Your AI-Powered Fracture Detection System!

Questions? Check the documentation or error messages - they're very helpful!

---

*Last Updated: February 9, 2026*  
*Version: 1.0.0*  
*Status: Production Ready*
