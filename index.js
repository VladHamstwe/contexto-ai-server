import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
app.use(express.json());
app.use(cors());

const client = new OpenAI({
  apiKey: process.env.OPENAI_KEY
});

async function getEmbedding(word) {
  const res = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: word
  });
  return res.data[0].embedding;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] ** 2;
    nb += b[i] ** 2;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

app.post("/similarity", async (req, res) => {
  const { secret, guess } = req.body;

  try {
    const v1 = await getEmbedding(secret);
    const v2 = await getEmbedding(guess);
    const sim = cosine(v1, v2);

    res.json({ similarity: sim });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => console.log("Server is running"));
