// generate_embeddings.js
// Usage: OPENAI_KEY=sk-... node generate_embeddings.js words.txt embeddings.jsonl
// Requirements: npm install openai p-limit

import fs from "fs";
import path from "path";
import OpenAI from "openai";
import pLimit from "p-limit";

const [,, wordsFile, outFile] = process.argv;
if (!wordsFile || !outFile) {
  console.error("Usage: node generate_embeddings.js words.txt embeddings.jsonl");
  process.exit(1);
}

const client = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const BATCH = 50;          // кількість слів в одному запиті (налаштуй)
const CONCURRENCY = 2;     // одночасних запитів (налаштуй під свої ліміти)
const MODEL = "text-embedding-3-large"; // або text-embedding-3-small

// Читаємо усі слова
const words = fs.readFileSync(wordsFile, "utf8")
  .split(/\r?\n/).map(s => s.trim()).filter(Boolean);

// Якщо файл виходу існує — зчитуємо вже збережені слова (resume)
const existing = new Set();
if (fs.existsSync(outFile)) {
  const lines = fs.readFileSync(outFile, "utf8").split(/\r?\n/).filter(Boolean);
  for (const l of lines) {
    try {
      const o = JSON.parse(l);
      existing.add(o.word);
    } catch {}
  }
}

console.log(`Total words: ${words.length}. Already saved: ${existing.size}`);

const limit = pLimit(CONCURRENCY);

async function getEmbeddingsBatch(batchWords) {
  // OpenAI embeddings supports multiple inputs at once
  const resp = await client.embeddings.create({
    model: MODEL,
    input: batchWords
  });
  return resp.data.map(d => d.embedding);
}

function normalize(vec) {
  let n = 0;
  for (let i=0;i<vec.length;i++) n += vec[i]*vec[i];
  n = Math.sqrt(n) || 1;
  const out = new Array(vec.length);
  for (let i=0;i<vec.length;i++) out[i] = vec[i]/n;
  return out;
}

(async () => {
  const outStream = fs.createWriteStream(outFile, { flags: "a" });

  for (let i=0;i<words.length;i+=BATCH) {
    const batch = words.slice(i, i+BATCH).filter(w => !existing.has(w));
    if (batch.length === 0) continue;

    // schedule call with concurrency control
    await limit(async () => {
      try {
        const embeds = await getEmbeddingsBatch(batch);
        for (let j=0;j<batch.length;j++) {
          const w = batch[j];
          const v = normalize(embeds[j]);
          const obj = { word: w, vector: v };
          outStream.write(JSON.stringify(obj) + "\n");
          existing.add(w);
        }
        console.log(`Saved ${Math.min(i+BATCH, words.length)}/${words.length}`);
      } catch (err) {
        console.error("Batch error:", err.message || err);
        // on error, wait a bit to avoid rate-limits
        await new Promise(r => setTimeout(r, 2000));
      }
    });
  }

  outStream.end();
  console.log("Done.");
})();
