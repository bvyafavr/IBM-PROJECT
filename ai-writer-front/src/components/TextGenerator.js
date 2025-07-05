// import React, { useState, useEffect } from "react";

// const TextGenerator = () => {
//   const [prompt, setPrompt] = useState("");
//   const [output, setOutput] = useState("");
//   const [loading, setLoading] = useState(false);
//   const [darkMode, setDarkMode] = useState(true);
//   const [detectionResult, setDetectionResult] = useState("");

//   useEffect(() => {
//     document.body.style.backgroundColor = darkMode ? "#0f0f1a" : "#ffffff";
//     document.body.style.color = darkMode ? "#f1f1f1" : "#1a1a1a";
//     document.body.style.transition = "background-color 0.3s ease, color 0.3s ease";
//   }, [darkMode]);
// const executeGenerate = async (customPrompt, mode) => {
//   setLoading(true);
//   setOutput("");
//   setDetectionResult("");

//   try {
//     const res = await fetch(
//       `${process.env.REACT_APP_API_URL || "http://localhost:4007"}/api/generate`,
//       {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ prompt: customPrompt, mode }),
//       }
//     );

//     const data = await res.json();

//     if (!data.result || typeof data.result !== "string") {
//       setOutput("❌ Error: No valid text returned from the API.");
//       return;
//     }

//     animateTyping(data.result);
//   } catch (err) {
//     console.error(err);
//     setOutput("❌ Error generating content.");
//   } finally {
//     setLoading(false);
//   }
// };


//   const handleMode = async (mode) => {
//     if (!prompt.trim()) return alert("⚠️ Please enter a prompt.");

//     if (mode === "story") {
//       const customPrompt = `Write a creative and engaging story based on this: "${prompt}"`;
//       executeGenerate(customPrompt);
//     } else if (mode === "humanise") {
//       setLoading(true);
//       setOutput("");
//       setDetectionResult("");
//       try {
//         const res = await fetch("http://localhost:5005/rewrite", {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ text: prompt }),
//         });

//         const data = await res.json();
//         if (!data.rewritten || typeof data.rewritten !== "string") {
//           setOutput("❌ Error: Pegasus-Gramformer returned invalid output.");
//           return;
//         }

//         animateTyping(data.rewritten);
//       } catch (err) {
//         console.error("Humanise API error:", err);
//         setOutput("❌ Error humanising text.");
//       } finally {
//         setLoading(false);
//       }
//     } else if (mode === "detect") {
//       handleDetect(prompt);
//     }
//   };

//   const handleDetect = async (textToCheck) => {
//     if (!textToCheck.trim()) return alert("⚠️ Paste or generate some content first.");
//     try {
//       setDetectionResult("🔍 Analyzing...");
//       const res = await fetch(
//         `${process.env.REACT_APP_API_URL || "http://localhost:4007"}/api/detect`,
//         {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ text: textToCheck }),
//         }
//       );
//       const data = await res.json();
//       setDetectionResult(data.result || "⚠️ Detection failed.");
//     } catch (err) {
//       console.error(err);
//       setDetectionResult("❌ Error during detection.");
//     }
//   };

//   const animateTyping = (text) => {
//     if (!text || typeof text !== "string") return;
//     let i = 0;
//     setOutput("");
//     const interval = setInterval(() => {
//       setOutput((prev) => prev + text.charAt(i));
//       i++;
//       if (i >= text.length) clearInterval(interval);
//     }, 15);
//   };

//   const styles = getStyles(darkMode);

//   return (
//     <div style={styles.wrapper}>
//       <div style={styles.header}>
//         <h1 style={styles.heading}>📝 AI Writing Assistant</h1>
//         <button onClick={() => setDarkMode(!darkMode)} style={styles.toggleBtn}>
//           {darkMode ? "🌞 Light Mode" : "🌙 Dark Mode"}
//         </button>
//       </div>

//       <textarea
//         placeholder="Enter your prompt..."
//         value={prompt}
//         onChange={(e) => setPrompt(e.target.value)}
//         style={styles.textarea}
//       />

//       <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginBottom: "20px" }}>
//         <button style={styles.button} onClick={() => handleMode("story")}>
//           🧙 Generate Story
//         </button>
//         <button style={styles.button} onClick={() => handleMode("humanise")}>
//           🧠 Humanise Text
//         </button>
//         <button style={styles.button} onClick={() => handleMode("detect")}>
//           🕵️ Detect AI/Plagiarism
//         </button>
//       </div>

//       {loading ? (
//         <p style={styles.loading}>⏳ Generating...</p>
//       ) : (
//         output && (
//           <div style={styles.outputBox}>
//             <h3>✨ Generated Output:</h3>
//             <p style={styles.output}>{output}</p>
//           </div>
//         )
//       )}

//       {detectionResult && (
//         <div style={{ marginTop: "20px", fontSize: "16px", fontWeight: "600", color: "#facc15" }}>
//           {detectionResult}
//         </div>
//       )}
//     </div>
//   );
// };

// const getStyles = (dark) => ({
//   wrapper: {
//     maxWidth: "750px",
//     margin: "50px auto",
//     padding: "35px",
//     borderRadius: "16px",
//     backgroundColor: dark ? "#1e1e2f" : "#ffffff",
//     boxShadow: "0 12px 25px rgba(0,0,0,0.2)",
//     transition: "0.4s ease-in-out",
//     fontFamily: "Inter, sans-serif",
//   },
//   header: {
//     display: "flex",
//     justifyContent: "space-between",
//     alignItems: "center",
//     marginBottom: "25px",
//   },
//   heading: {
//     fontSize: "28px",
//     fontWeight: "bold",
//   },
//   toggleBtn: {
//     padding: "8px 16px",
//     borderRadius: "8px",
//     border: "none",
//     backgroundColor: dark ? "#2a2a3c" : "#e2e2e2",
//     color: dark ? "#f1f1f1" : "#333",
//     cursor: "pointer",
//     fontWeight: "600",
//     transition: "0.3s",
//   },
//   textarea: {
//     width: "100%",
//     height: "140px",
//     padding: "14px",
//     borderRadius: "12px",
//     fontSize: "16px",
//     backgroundColor: dark ? "#2c2c3b" : "#fff",
//     color: dark ? "#fff" : "#000",
//     border: `1px solid ${dark ? "#444" : "#ccc"}`,
//     marginBottom: "20px",
//     resize: "vertical",
//   },
//   button: {
//     padding: "14px",
//     fontSize: "16px",
//     borderRadius: "10px",
//     background: "linear-gradient(to right, #6a11cb, #2575fc)",
//     color: "#fff",
//     border: "none",
//     cursor: "pointer",
//     fontWeight: "600",
//     transition: "transform 0.2s ease",
//   },
//   loading: {
//     textAlign: "center",
//     fontSize: "18px",
//     marginBottom: "30px",
//     color: dark ? "#ccc" : "#666",
//   },
//   outputBox: {
//     background: dark
//       ? "linear-gradient(135deg, #1a1a2e, #2e2e4d)"
//       : "linear-gradient(135deg, #ffffff, #f7f7f7)",
//     padding: "20px",
//     borderRadius: "12px",
//     border: `1px solid ${dark ? "#555" : "#ccc"}`,
//     boxShadow: dark ? "0 0 12px rgba(255,255,255,0.05)" : "0 0 12px rgba(0,0,0,0.05)",
//   },
//   output: {
//     whiteSpace: "pre-wrap",
//     lineHeight: "1.6",
//     fontSize: "16px",
//     color: dark ? "#eaeaea" : "#222",
//     marginTop: "12px",
//   },
// });

// export default TextGenerator;



import React, { useState, useEffect } from "react";

const TextGenerator = () => {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [detectionResult, setDetectionResult] = useState("");

  useEffect(() => {
    document.body.style.backgroundColor = darkMode ? "#0f0f1a" : "#ffffff";
    document.body.style.color = darkMode ? "#f1f1f1" : "#1a1a1a";
    document.body.style.transition = "background-color 0.3s ease, color 0.3s ease";
  }, [darkMode]);

  const executeGenerate = async (customPrompt, mode) => {
    setLoading(true);
    setOutput("");
    setDetectionResult("");

    try {
      const res = await fetch(
        `${process.env.REACT_APP_API_URL || "http://localhost:4007"}/api/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: customPrompt, mode }),
        }
      );

      const data = await res.json();

      if (!data.result || typeof data.result !== "string") {
        setOutput("❌ Error: No valid text returned from the API.");
        return;
      }

      animateTyping(data.result);
    } catch (err) {
      console.error(err);
      setOutput("❌ Error generating content.");
    } finally {
      setLoading(false);
    }
  };

  const handleMode = async (mode) => {
    if (!prompt.trim()) return alert("⚠️ Please enter a prompt.");

    if (mode === "story" || mode === "humanise") {
      executeGenerate(prompt, mode);
    } else if (mode === "detect") {
      handleDetect(prompt);
    }
  };

  const handleDetect = async (textToCheck) => {
    if (!textToCheck.trim()) return alert("⚠️ Paste or generate some content first.");
    try {
      setDetectionResult("🔍 Analyzing...");
      const res = await fetch(
        `${process.env.REACT_APP_API_URL || "http://localhost:4007"}/api/detect`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textToCheck }),
        }
      );
      const data = await res.json();
      setDetectionResult(data.result || "⚠️ Detection failed.");
    } catch (err) {
      console.error(err);
      setDetectionResult("❌ Error during detection.");
    }
  };

  const animateTyping = (text) => {
    if (!text || typeof text !== "string") return;
    let i = 0;
    setOutput("");
    const interval = setInterval(() => {
      setOutput((prev) => prev + text.charAt(i));
      i++;
      if (i >= text.length) clearInterval(interval);
    }, 15);
  };

  const styles = getStyles(darkMode);

  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <h1 style={styles.heading}>📝 AI Writing Assistant</h1>
        <button onClick={() => setDarkMode(!darkMode)} style={styles.toggleBtn}>
          {darkMode ? "🌞 Light Mode" : "🌙 Dark Mode"}
        </button>
      </div>

      <textarea
        placeholder="Enter your prompt..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        style={styles.textarea}
      />

      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginBottom: "20px" }}>
        <button style={styles.button} onClick={() => handleMode("story")}>
          🧙 Generate Story
        </button>
        <button style={styles.button} onClick={() => handleMode("humanise")}>
          🧠 Humanise Text
        </button>
        <button style={styles.button} onClick={() => handleMode("detect")}>
          🕵️ Detect AI/Plagiarism
        </button>
      </div>

      {loading ? (
        <p style={styles.loading}>⏳ Generating...</p>
      ) : (
        output && (
          <div style={styles.outputBox}>
            <h3>✨ Generated Output:</h3>
            <p style={styles.output}>{output}</p>
          </div>
        )
      )}

      {detectionResult && (
        <div style={{ marginTop: "20px", fontSize: "16px", fontWeight: "600", color: "#facc15" }}>
          {detectionResult}
        </div>
      )}
    </div>
  );
};

const getStyles = (dark) => ({
  wrapper: {
    maxWidth: "750px",
    margin: "50px auto",
    padding: "35px",
    borderRadius: "16px",
    backgroundColor: dark ? "#1e1e2f" : "#ffffff",
    boxShadow: "0 12px 25px rgba(0,0,0,0.2)",
    transition: "0.4s ease-in-out",
    fontFamily: "Inter, sans-serif",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "25px",
  },
  heading: {
    fontSize: "28px",
    fontWeight: "bold",
  },
  toggleBtn: {
    padding: "8px 16px",
    borderRadius: "8px",
    border: "none",
    backgroundColor: dark ? "#2a2a3c" : "#e2e2e2",
    color: dark ? "#f1f1f1" : "#333",
    cursor: "pointer",
    fontWeight: "600",
    transition: "0.3s",
  },
  textarea: {
    width: "100%",
    height: "140px",
    padding: "14px",
    borderRadius: "12px",
    fontSize: "16px",
    backgroundColor: dark ? "#2c2c3b" : "#fff",
    color: dark ? "#fff" : "#000",
    border: `1px solid ${dark ? "#444" : "#ccc"}`,
    marginBottom: "20px",
    resize: "vertical",
  },
  button: {
    padding: "14px",
    fontSize: "16px",
    borderRadius: "10px",
    background: "linear-gradient(to right, #6a11cb, #2575fc)",
    color: "#fff",
    border: "none",
    cursor: "pointer",
    fontWeight: "600",
    transition: "transform 0.2s ease",
  },
  loading: {
    textAlign: "center",
    fontSize: "18px",
    marginBottom: "30px",
    color: dark ? "#ccc" : "#666",
  },
  outputBox: {
    background: dark
      ? "linear-gradient(135deg, #1a1a2e, #2e2e4d)"
      : "linear-gradient(135deg, #ffffff, #f7f7f7)",
    padding: "20px",
    borderRadius: "12px",
    border: `1px solid ${dark ? "#555" : "#ccc"}`,
    boxShadow: dark ? "0 0 12px rgba(255,255,255,0.05)" : "0 0 12px rgba(0,0,0,0.05)",
  },
  output: {
    whiteSpace: "pre-wrap",
    lineHeight: "1.6",
    fontSize: "16px",
    color: dark ? "#eaeaea" : "#222",
    marginTop: "12px",
  },
});

export default TextGenerator;

