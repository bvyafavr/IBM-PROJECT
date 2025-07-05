const express = require("express");
const router = express.Router();
const { GoogleGenerativeAI } = require("@google/generative-ai");

require("dotenv").config();
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

router.post("/", async (req, res) => {
  const { text } = req.body;

  if (!text || typeof text !== "string" || !text.trim()) {
    return res.status(400).json({ error: "Text is required for detection." });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

    const detectionPrompt = `
You are an AI content detection expert. 
Analyze the following text and clearly tell whether it is AI-generated or human-written.
Also state if it's likely to get flagged by tools like Turnitin, GPTZero, or Originality.ai.
Be concise and direct in under 3 lines.

Text:
"""${text}"""
    `;

    const result = await model.generateContent({
      contents: [{ parts: [{ text: detectionPrompt }] }],
    });

    const response = await result.response;
    const verdict = (await response.text()).trim(); // ✅ Fix here

    if (!verdict) {
      return res.status(500).json({ error: "Detection failed." });
    }

    res.json({ result: verdict });
  } catch (error) {
    console.error("Detection Error:", error.message);
    res.status(500).json({ error: "Error during detection." });
  }
});

module.exports = router;
