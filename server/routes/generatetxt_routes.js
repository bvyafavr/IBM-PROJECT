// const express = require("express");
// const router = express.Router();
// require("dotenv").config();
// const { GoogleGenerativeAI } = require("@google/generative-ai");

// const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// router.post("/", async (req, res) => {
//   const { prompt } = req.body; // ← THIS MUST BE "prompt"

//   if (!prompt || typeof prompt !== "string" || !prompt.trim()) {
//     return res.status(400).json({ error: "Prompt is required." });
//   }

//   try {
//     const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

//     const result = await model.generateContent({
//       contents: [{ parts: [{ text: prompt }] }],
//     });

//     const response = await result.response;
//     const candidates = response.candidates;
//     const raw = candidates?.[0]?.content?.parts?.[0]?.text ?? "";
//     const text = raw.trim();

//     if (!text) {
//       return res.status(500).json({ error: "No valid text returned." });
//     }

//     res.json({ result: text });
//   } catch (error) {
//     console.error("Gemini API Error:", error);
//     res.status(500).json({ error: "Failed to generate content." });
//   }
// });

// module.exports = router;



// const express = require("express");
// const router = express.Router();
// require("dotenv").config();
// const { GoogleGenerativeAI } = require("@google/generative-ai");
// const axios = require("axios"); // ← Add axios

// const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// router.post("/", async (req, res) => {
//   const { prompt } = req.body;

//   if (!prompt || typeof prompt !== "string" || !prompt.trim()) {
//     return res.status(400).json({ error: "Prompt is required." });
//   }

//   try {
//     const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

//     const result = await model.generateContent({
//       contents: [{ parts: [{ text: prompt }] }],
//     });

//     const response = await result.response;
//     const candidates = response.candidates;
//     const raw = candidates?.[0]?.content?.parts?.[0]?.text ?? "";
//     const text = raw.trim();

//     if (!text) {
//       return res.status(500).json({ error: "No valid text returned." });
//     }

//     // 🔁 Send to FastAPI Pegasus server
//     const humanized = await axios.post("http://localhost:5005/rewrite", {
//       text,
//     });

//     res.json({ result: humanized.data.rewritten });
//   } catch (error) {
//     console.error("Gemini or Pegasus API Error:", error.message);
//     res.status(500).json({ error: "Failed to generate or humanize content." });
//   }
// });

// module.exports = router;


// const express = require("express");
// const router = express.Router();
// require("dotenv").config();
// const axios = require("axios");
// const { GoogleGenerativeAI } = require("@google/generative-ai");

// const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// router.post("/", async (req, res) => {
//   const { prompt } = req.body;

//   if (!prompt || typeof prompt !== "string" || !prompt.trim()) {
//     return res.status(400).json({ error: "Prompt is required." });
//   }

//   try {
//     // Step 1: Send text to Pegasus + T5 API
//     const pegasusResponse = await axios.post("http://localhost:5005/rewrite", {
//       text: prompt,
//     });

//     const rewritten = pegasusResponse.data?.rewritten?.trim();

//     if (!rewritten) {
//       return res.status(500).json({ error: "Paraphrasing failed." });
//     }

//     // Step 2: Ask Gemini to only fix grammar and punctuation
//     const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
//     const grammarPrompt = `Correct only the grammar and punctuation in the following text. Do not change any words or structure:\n\n"""${rewritten}"""`;

//     const result = await model.generateContent({
//       contents: [{ parts: [{ text: grammarPrompt }] }],
//     });

//     const response = await result.response;
//     const cleanText = response?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();

//     if (!cleanText) {
//       return res.status(500).json({ error: "No valid text returned from Gemini." });
//     }

//     res.json({ result: cleanText });
//   } catch (error) {
//     console.error("Humanisation error:", error.message);
//     res.status(500).json({ error: "Something went wrong during humanisation." });
//   }
// });

// module.exports = router;

const express = require("express");
const router = express.Router();
require("dotenv").config();
const { GoogleGenerativeAI } = require("@google/generative-ai");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

router.post("/", async (req, res) => {
  const { prompt, mode } = req.body;

  if (!prompt || typeof prompt !== "string" || !prompt.trim()) {
    return res.status(400).json({ error: "Prompt is required." });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

    let finalPrompt = "";

    if (mode === "story") {
      finalPrompt = `Write a creative and engaging story based on this: "${prompt}"`;
    } else if (mode === "humanise") {
      finalPrompt = `Rewrite the following text in simple, natural English as if written by a human. Don't make it fancy or poetic:\n\n"${prompt}"`;
    } else {
      return res.status(400).json({ error: "Unsupported mode." });
    }

    const result = await model.generateContent({
      contents: [{ parts: [{ text: finalPrompt }] }],
    });

    const response = await result.response;
    const raw = response?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();

    if (!raw) {
      return res.status(500).json({ error: "No valid text returned from Gemini." });
    }

    res.json({ result: raw });
  } catch (err) {
    console.error("Gemini error:", err.message);
    res.status(500).json({ error: "Something went wrong with Gemini API." });
  }
});

module.exports = router;


