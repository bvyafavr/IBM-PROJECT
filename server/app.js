const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

// ✅ Generate route
const generate_Routes = require("./routes/generatetxt_routes");
app.use("/api/generate", generate_Routes);

// ✅ Detect route
const detect_Routes = require("./routes/detect");
app.use("/api/detect", detect_Routes);

// ✅ Pegasus route
const pegasus_Routes = require("./routes/pegasus");
app.use("/api/pegasus", pegasus_Routes);

// ✅ Root route
app.get("/", (req, res) => {
  res.send("🔥 API is working");
});

module.exports = app;
