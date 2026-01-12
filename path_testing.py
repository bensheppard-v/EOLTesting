python -c "import sys
try: import tkinter; print('tkinter OK')
except Exception as e: print('tkinter error', e)
try: import PyQt5; print('PyQt5 OK')
except Exception as e: print('PyQt5 error', e)
print('python:', sys.executable)"