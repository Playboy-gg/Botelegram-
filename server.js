import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ----- Config -----
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
const DEFAULT_MODEL = process.env.GEMINI_MODEL || 'gemini-1.5-flash';
const API_KEY = process.env.GEMINI_API_KEY || '';

// Create SDK client lazily so app can boot without key, but fail on use
let genAI = null;
function getGenAI() {
  if (!API_KEY) {
    throw new Error('Missing GEMINI_API_KEY in environment');
  }
  if (!genAI) {
    genAI = new GoogleGenerativeAI(API_KEY);
  }
  return genAI;
}

// ----- App -----
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '1mb' }));

// Serve static UI
const publicDir = path.join(__dirname, 'public');
app.use(express.static(publicDir));
app.get('/', (req, res) => {
  res.sendFile(path.join(publicDir, 'chat.html'));
});

// In-memory sessions: { [sessionId]: { history: Array<HistoryMessage> } }
// HistoryMessage: { role: 'user'|'model', parts: [{ text: string }] }
const sessionStore = new Map();
const MAX_TURNS = 50;

function ensureSession(sessionId) {
  if (!sessionStore.has(sessionId)) {
    sessionStore.set(sessionId, { history: [] });
  }
  return sessionStore.get(sessionId);
}

function clampTurns(history) {
  // Keep only the most recent MAX_TURNS messages
  const over = history.length - MAX_TURNS * 2; // user+model per turn
  if (over > 0) {
    return history.slice(over);
  }
  return history;
}

// Health endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', modelDefault: DEFAULT_MODEL, hasApiKey: Boolean(API_KEY) });
});

// Reset a chat session
app.post('/api/session/reset', (req, res) => {
  const { sessionId } = req.body || {};
  if (!sessionId) return res.status(400).json({ error: 'Missing sessionId' });
  sessionStore.delete(sessionId);
  res.json({ ok: true });
});

// Streaming chat endpoint (NDJSON streaming)
app.post('/api/chat', async (req, res) => {
  try {
    const {
      sessionId,
      message,
      model: modelName,
      systemPrompt,
      temperature,
      maxOutputTokens
    } = req.body || {};

    if (!sessionId) {
      return res.status(400).json({ error: 'Missing sessionId' });
    }
    if (typeof message !== 'string' || message.trim() === '') {
      return res.status(400).json({ error: 'Missing message' });
    }

    // Prepare response for streaming NDJSON
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const session = ensureSession(sessionId);
    session.history = clampTurns(session.history);

    const gen = getGenAI();
    const model = gen.getGenerativeModel({
      model: modelName || DEFAULT_MODEL,
      ...(systemPrompt ? { systemInstruction: systemPrompt } : {})
    });

    const generationConfig = {};
    if (typeof temperature === 'number') generationConfig.temperature = temperature;
    if (typeof maxOutputTokens === 'number') generationConfig.maxOutputTokens = maxOutputTokens;

    const chat = model.startChat({
      history: session.history,
      ...(Object.keys(generationConfig).length ? { generationConfig } : {})
    });

    // Append the user message locally after starting stream
    const userMessage = { role: 'user', parts: [{ text: message }] };
    const assistantMessage = { role: 'model', parts: [{ text: '' }] };

    let aborted = false;
    const onClose = () => { aborted = true; };
    req.on('close', onClose);

    let aggregated = '';
    try {
      const result = await chat.sendMessageStream(message);
      for await (const chunk of result.stream) {
        if (aborted) break;
        const chunkText = chunk?.text ? chunk.text() : '';
        if (chunkText) {
          aggregated += chunkText;
          res.write(JSON.stringify({ type: 'content', text: chunkText }) + '\n');
        }
      }
      // Ensure final response
      const final = result?.response?.text ? result.response.text() : aggregated;
      aggregated = final || aggregated;
    } catch (err) {
      // Emit an error frame then end
      res.write(JSON.stringify({ type: 'error', error: (err && err.message) || 'Generation failed' }) + '\n');
    } finally {
      req.off('close', onClose);
    }

    // Commit to history if not aborted
    if (!aborted && aggregated) {
      assistantMessage.parts[0].text = aggregated;
      session.history.push(userMessage);
      session.history.push(assistantMessage);
    }

    res.write(JSON.stringify({ type: 'done' }) + '\n');
    res.end();
  } catch (e) {
    const status = e.message.includes('GEMINI_API_KEY') ? 500 : 400;
    res.status(status).json({ error: e.message || 'Unexpected error' });
  }
});

app.listen(PORT, () => {
  console.log(`Gemini Chatbot server listening on http://localhost:${PORT}`);
});
