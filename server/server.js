// Load environment variables
require("dotenv").config();

const express = require("express");
const app = require("./app");

const PORT = process.env.PORT || 4007;

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
