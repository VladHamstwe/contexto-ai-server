import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(bodyParser.json());

const client = new OpenAI({
    apiKey: process.env.OPENAI_KEY
});

// --- SECRET WORD ----
const SECRET_WORD = "apple"; // хочеш — зроблю випадковим

// --- КЕШ ДЛЯ ВЕКТОРІВ ---
const embedCache = {};

// Створення embedding
async function getEmbedding(word) {
    if (embedCache[word]) return embedCache[word];

    const response = await client.embeddings.create({
        model: "text-embedding-3-small",
        input: word
    });

    const vector = response.data[0].embedding;
    embedCache[word] = vector;
    return vector;
}

// Косинусна схожість
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// --- Аналіз здогадки ---
app.post("/guess", async (req, res) => {
    try {
        const guess = req.body.word.toLowerCase();

        const secretVector = await getEmbedding(SECRET_WORD);
        const guessVector = await getEmbedding(guess);

        const similarity = cosineSimilarity(secretVector, guessVector);

        // Перетворюємо схожість на "позицію" як у Contexto
        const position = Math.floor(1 / (similarity || 0.000001));

        res.json({
            word: guess,
            similarity,
            position
        });

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Server error" });
    }
});

// --- health check ---
app.get("/", (req, res) => {
    res.send("AI server is running");
});

app.listen(3000, () => {
    console.log("Server running on port 3000");
});

