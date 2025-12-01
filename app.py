import os
import json
from flask import Flask, render_template, request
from predictor import Predictor

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load model và class names ngay trong thư mục webapp
MODEL_PATH = "model_final.h5"
predictor = Predictor(MODEL_PATH)

with open("class_names.json", "r") as f:
    class_names = json.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img = request.files.get("image")
        if img:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(img_path)

            result = predictor.predict(img_path, class_names)

            return render_template(
                "result.html",
                img_path=img_path,
                label=result["label"],
                prob=round(result["prob"] * 100, 2)
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
