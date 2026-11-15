"""CI smoke test: the committed model must load and the Streamlit app must run on it."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("superkart_project/deployment/app.py", default_timeout=120).run()
assert not at.exception, at.exception
at.button[0].click().run()
assert not at.exception, at.exception
assert at.success, "no prediction was displayed"
print("Smoke test passed ->", at.success[0].value)
