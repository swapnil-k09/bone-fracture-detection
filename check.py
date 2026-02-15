import os

print("Current directory:", os.getcwd())
print("\nFiles here:")
for item in os.listdir('.'):
    print(f"  - {item}")

if os.path.exists('app.py'):
    print("\n✅ app.py found!")
else:
    print("\n❌ app.py NOT found - you're in the wrong folder!")

if os.path.exists('templates'):
    print("✅ templates folder found!")
    print("   Templates:")
    for t in os.listdir('templates'):
        print(f"      - {t}")
else:
    print("❌ templates folder NOT found!")
