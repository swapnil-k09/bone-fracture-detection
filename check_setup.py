"""
Diagnostic script to check Flask setup
"""
import os

print("=" * 60)
print("FLASK APP DIAGNOSTIC")
print("=" * 60)

# Check current directory
print(f"\n1. Current Directory: {os.getcwd()}")

# Check if app.py exists
app_exists = os.path.exists('app.py')
print(f"2. app.py exists here: {app_exists}")

# Check if templates folder exists
templates_exists = os.path.exists('templates')
print(f"3. templates/ folder exists: {templates_exists}")

# Check template files
if templates_exists:
    templates = os.listdir('templates')
    print(f"4. Template files found: {len(templates)}")
    for t in sorted(templates):
        print(f"   - {t}")
else:
    print("4. No templates folder!")

# Check if privacy.html exists
privacy_exists = os.path.exists('templates/privacy.html')
terms_exists = os.path.exists('templates/terms.html')
print(f"\n5. privacy.html exists: {privacy_exists}")
print(f"6. terms.html exists: {terms_exists}")

# Check app.py for routes
if app_exists:
    with open('app.py', 'r') as f:
        content = f.read()
        has_privacy = 'def privacy' in content
        has_terms = 'def terms' in content
        print(f"\n7. app.py has privacy route: {has_privacy}")
        print(f"8. app.py has terms route: {has_terms}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

if app_exists and templates_exists and privacy_exists and terms_exists:
    print("✅ All files are in place!")
    print("✅ Routes should work after restarting Flask")
    print("\nTo restart Flask:")
    print("1. Press Ctrl+C to stop")
    print("2. Run: python app.py")
    print("3. Try: http://localhost:5000/privacy")
else:
    print("❌ Missing files detected!")
    if not app_exists:
        print("   - app.py not found in current directory")
    if not templates_exists:
        print("   - templates/ folder not found")
    if not privacy_exists:
        print("   - privacy.html not found")
    if not terms_exists:
        print("   - terms.html not found")
    print("\n⚠️  You may be in the wrong directory!")
    print("   Navigate to: bone_fracture_detection/")

print("=" * 60)
