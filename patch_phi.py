"""Patch script v2: directly target the exact substring"""
with open('generate_notebooks.py', 'rb') as f:
    raw = f.read()

# The exact bytes in the file for the problematic line
old_bytes = b'ax.set_ylabel(\\"Learned Activation Response $\\\\\\\\phi(x)$\\", fontsize=11)'
new_bytes = b'ax.set_ylabel(\\"Learned KAN Activation Response\\", fontsize=11)'

if old_bytes in raw:
    raw = raw.replace(old_bytes, new_bytes)
    print("Replaced phi escape (bytes match).")
else:
    # Try unicode
    content = raw.decode('utf-8')
    target = 'ax.set_ylabel(\\"Learned Activation Response $\\\\\\\\phi(x)$\\", fontsize=11)'
    print("Target found:", target in content)
    idx = content.find("Learned Activation")
    if idx >= 0:
        print("Context:", repr(content[idx-2:idx+90]))

with open('generate_notebooks.py', 'wb') as f:
    f.write(raw)
print("Done.")
