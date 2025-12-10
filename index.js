// index.js
import express from "express";
import fs from "fs";
import cors from "cors";
import bodyParser from "body-parser";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(bodyParser.json());

const client = new OpenAI({ apiKey: process.env.OPENAI_KEY });

// Конфіг
const EMB_FILE = process.env.EMB_FILE || "./embeddings.jsonl";
const MODEL = process.env.EMBED_MODEL || "text-embedding-3-large";
const PORT = process.env.PORT || 3000;

// Завантажуємо embeddings в пам'ять (words[], vectorsFlattened, dims)
console.log("Loading embeddings...");
const lines = fs.readFileSync(EMB_FILE, "utf8").split(/\r?\n/).filter(Boolean);
const words = [];
let vectors = null; // Float32Array flattened
let dims = null;
for (let idx=0; idx<lines.length; idx++) {
  const obj = JSON.parse(lines[idx]);
  if (idx === 0) {
    dims = obj.vector.length;
    vectors = new Float32Array(lines.length * dims);
  }
  words.push(obj.word);
  for (let j=0;j<dims;j++) {
    vectors[idx*dims + j] = obj.vector[j];
  }
}
console.log(`Loaded ${words.length} words, dims=${dims}`);

function dotProductGuess(idx, guessVec) {
  let s = 0.0;
  const base = idx * dims;
  for (let i=0;i<dims;i++) s += vectors[base + i] * guessVec[i];
  return s;
}

// Політика: для сервісу — повертаємо позицію та топN (наприклад top 20)
app.post("/guess", async (req, res) => {
  try {
    const guessRaw = (req.body.word || "").toString().trim().toLowerCase();
    if (!guessRaw) return res.status(400).json({ error: "No word" });

    // отримуємо embedding для здогадки та нормалізуємо
    const embResp = await client.embeddings.create({
      model: MODEL,
      input: guessRaw
    });
    let guessVec = embResp.data[0].embedding;
    // normalize
    let n = 0;
    for (let i=0;i<guessVec.length;i++) n += guessVec[i]*guessVec[i];
    n = Math.sqrt(n)||1;
    for (let i=0;i<guessVec.length;i++) guessVec[i] = guessVec[i]/n;

    // якщо виміри не співпадають з завантаженими — помилка
    if (guessVec.length !== dims) {
      return res.status(500).json({ error: "Dimension mismatch" });
    }

    // обчислюємо dot product з усіма векторами
    // Збираємо топN (heap або простий варіант)
    const topN = parseInt(req.query.top || "20", 10);
    const sims = new Array(words.length);
    for (let i=0;i<words.length;i++) {
      sims[i] = dotProductGuess(i, guessVec);
    }

    // знаходимо позицію відсортовану descending (1 — найвищий similarity)
    // Для ефективності: можна знайти rank без повного сортування. Але тут сортуємо індекси.
    const idxs = sims.map((s,i) => ({i, s}));
    idxs.sort((a,b) => b.s - a.s); // від найбільшого (1) до найменшого

    // знаходимо позицію першого появи слова == guessRaw (якщо слово є в dictionary)
    let position = -1;
    for (let rank=0; rank<idxs.length; rank++) {
      if (words[idxs[rank].i] === guessRaw) { position = rank+1; break; }
    }

    // формуємо topN
    const top = [];
    for (let k=0;k<Math.min(topN, idxs.length);k++) {
      top.push({
        word: words[idxs[k].i],
        similarity: idxs[k].s,
        rank: k+1
      });
    }

    // якщо слово не в базі — позиція = місце за ранжуванням серед усіх (розрахуємо на основі similarity)
    if (position === -1) {
      // знаходимо rank куди потрапило б слово (за similarity)
      // знайдемо перше місце де sims[rank].s <= similarity
      let pos = 1;
      while (pos <= idxs.length && idxs[pos-1].s > (dot=0,0)) pos++;
      // але краще: знайти rank за бинарним пошуком по sim; ми маємо sims array...
      // Простіше — знаходимо місце, де similarity перевищує; зробимо цикл
      let p = 1;
      while (p <= idxs.length && idxs[p-1].s > top[top.length-1].similarity) p++;
      position = idxs.findIndex(x => x.s <= top[top.length-1].similarity) + 1;
      if (position === 0) position = idxs.length;
    }

    // Повертаємо similarity щодо топ1 (секретного слова може бути будь-яким)
    return res.json({
      word: guessRaw,
      position,
      top
    });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message || err });
  }
});

app.get("/", (req, res) => res.send("AI server is running"));

app.listen(PORT, () => console.log("Server running on port", PORT));
