document.addEventListener("DOMContentLoaded", function () {
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const textInput = document.getElementById("textInput");
  const resultDiv = document.getElementById("result");
  const labelDiv = document.getElementById("label");
  const scoreDiv = document.getElementById("score");
  const confidenceBar = document.getElementById("confidenceBar");

  const sentimentEmoji = {
    positive: "😊",
    negative: "😡",
    neutral: "😐"
  };

  predictBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (!text) {
      alert("Please enter some text.");
      return;
    }

    predictBtn.disabled = true;
    predictBtn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Analyzing...`;

    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });

      const data = await resp.json();

      if (!resp.ok) {
        alert(data.error || "Prediction failed.");
      } else {
        resultDiv.style.display = "block";
        resultDiv.classList.add("fade-in");

        const sentiment = data.label.toLowerCase();
        const confidence = (data.score * 100).toFixed(2);

        labelDiv.innerHTML = `${sentiment.toUpperCase()} ${sentimentEmoji[sentiment] || ""}`;
        labelDiv.className =
          sentiment === "positive"
            ? "sentiment-positive"
            : sentiment === "negative"
            ? "sentiment-negative"
            : "sentiment-neutral";

        scoreDiv.innerHTML = `Confidence: <b>${confidence}%</b>`;
        confidenceBar.style.width = `${confidence}%`;
        confidenceBar.innerText = `${confidence}%`;
      }
    } catch (err) {
      alert("Request failed: " + err.message);
      console.error(err);
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerText = "Predict";
    }
  });

  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    resultDiv.style.display = "none";
  });
});
