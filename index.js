const express = require("express");
const bodyParser = require("body-parser");

const app = express();
app.use(bodyParser.json());

// entry route
app.get("/", (req, res) => {
  res.send("API is running");
});

const PORT = 5000;
app.listen(PORT, () =>
  console.log(`Server is running on http://localhost:${PORT}`)
);

// to start another process like python script
const { spawn } = require("child_process");

// endpoint
app.post("/predict", (req, res) => {
  const inputData = req.body;

  // calling python script
  const python = spawn("python", [
    "heart_load_save.py",
    JSON.stringify(inputData),
  ]);

  // whatever python prints using print() and captures the predictions
  python.stdout.on("data", (data) => {
    const output = data.toString().trim();

    // Try parsing Python output as JSON, fallback to plain string
    try {
      const parsed = JSON.parse(output);
      res.json(parsed); // Sends proper JSON with Content-Type: application/json
    } catch (e) {
      res.json({ prediction: output });
    }
  });
});
