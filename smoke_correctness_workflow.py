"""Small Flask-test-client smoke workflow for correctness-critical routes.

This script exercises upload, one committed Apply, Final Clean, and file
existence checks without browser automation. It is a fast regression signal for
CSRF and persistence plumbing, not a replacement for rendered Playwright tests.
"""

import io
import os

os.environ.setdefault("INDATAOUT_TRUSTED_LOCAL", "1")

import app


client = app.app.test_client()

client.get("/")
with client.session_transaction() as sess:
    csrf_token = sess.get("_csrf_token")

upload = client.post(
    "/upload",
    data={"csrf_token": csrf_token, "file": (io.BytesIO(b"x,y\n a ,1\nb,2\n"), "sample.csv")},
    content_type="multipart/form-data",
)
print("upload", upload.status_code)

with client.session_transaction() as sess:
    sid = sess.get("upload_sid")
    filename = sess.get("upload_filename")

print("sid", bool(sid), "filename", filename)

apply_resp = client.post(
    "/preview_clean",
    data={
        "filename": filename,
        "csrf_token": csrf_token,
        "persist_apply": "1",
        "preview_scope": "section",
        "trim_whitespace": "1",
        "remove_duplicates": "0",
        "drop_empty_columns": "0",
        "drop_constant_columns": "0",
        "convert_dtypes": "0",
        "missing_strategy": "ignore",
    },
)
apply_json = apply_resp.get_json() or {}
print("apply", apply_resp.status_code, apply_json.get("success"), apply_json.get("summary", {}).get("rows_after"))

clean = client.post("/clean", data={"filename": filename, "csrf_token": csrf_token})
print("clean", clean.status_code)
print("original_exists", os.path.exists(os.path.join(app.UPLOAD_FOLDER, sid, filename)))

with client.session_transaction() as sess:
    cleaned_filename = sess.get("cleaned_filename")

cleaned_path = os.path.join(app.UPLOAD_FOLDER, sid, "cleaned", cleaned_filename)
print("cleaned_exists", os.path.exists(cleaned_path), cleaned_filename)
