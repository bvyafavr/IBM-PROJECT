const express = require("express");
const router = express.Router();
const axios = require("axios");
const { GoogleGenerativeAI } = require("@google/generative-ai");
require("dotenv").config();

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

router.post("/", async (req, res) => {
  try {
    const { text } = req.body;

    // 1. Pegasus & T5 rewrite
    const pegasusRes = await axios.post("http://localhost:5005/rewrite", {
      text,
    });

    const rewritten = pegasusRes.data?.rewritten?.trim();
    if (!rewritten) {
      return res.status(500).json({ error: "Pegasus failed to rewrite." });
    }

    // 2. Send to Gemini for final grammar & punctuation fix
    const grammarPrompt = `Correct only the grammar and punctuation in this text. Don't change the structure or meaning:\n\n"""${rewritten}"""`;
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

    const result = await model.generateContent({
      contents: [{ parts: [{ text: grammarPrompt }] }],
    });

    const response = await result.response;
    const finalText = response?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();

    if (!finalText) {
      return res.status(500).json({ error: "Gemini failed to return text." });
    }

    res.json({ rewritten: finalText });
  } catch (error) {
    console.error("Pegasus+Gemini error:", error.message);
    res.status(500).json({ error: "Failed to rewrite using Pegasus+Gemini" });
  }
});

module.exports = router;
