// backend/server.js
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';


const app = express();

// ✅ Localhost default
const PORT = 5000;

// ✅ Resolve __dirname properly
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ✅ Ensure folders exist
const uploadDir = path.join(__dirname, 'uploads');
const resultsDir = path.join(__dirname, 'results');
const preloadedDir = path.join(__dirname, 'preloaded_datasets'); // NEW

if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir);
if (!fs.existsSync(preloadedDir)) fs.mkdirSync(preloadedDir); // NEW

// --- Middleware ---
app.use(cors());
app.use(express.json());

// --- Serve static frontend ---
app.use(express.static(path.join(__dirname, '../frontend')));
app.use('/results', express.static(resultsDir));

// --- File Upload (Multer) ---
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const upload = multer({ storage });

// --- Simple Test Route ---
app.get('/ping', (req, res) => res.json({ message: 'Server is running locally ✅' }));

function runPythonAnalysis(filePath, res, deleteFileAfter = false) {
    const modelPath = path.join(__dirname, 'model.py');
    console.log(`-> Running Python script on: ${filePath}`);

    // Check if file exists first
    if (!fs.existsSync(filePath)) {
        console.error(`-> File not found: ${filePath}`);
        return res.status(404).json({ error: 'Dataset file not found on server.' });
    }

    const pythonProcess = spawn('python', [modelPath, filePath]);
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
        console.log(`[Python STDOUT]: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Python STDERR]: ${data}`);
        pythonError += data.toString();
    });

    pythonProcess.on('close', (code) => {
        console.log(`-> Python exited with code ${code}`);

        if (deleteFileAfter) {
            fs.unlink(filePath, (err) => {
                if (err) console.error(`-> Failed to delete uploaded file: ${err}`);
                else console.log(`-> Deleted uploaded file: ${filePath}`);
            });
        }

        if (code !== 0) {
            return res.status(500).json({
                error: 'An error occurred during model analysis.',
                details: pythonError || 'Unknown Python error.'
            });
        }

        try {
            const resultsPath = path.join(resultsDir, 'metrics.json');
            const data = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
            console.log('-> Sending results to client.');
            res.json(data);
        } catch (err) {
            console.error('-> Failed to read results file.', err);
            res.status(500).json({
                error: 'Failed to read analysis results.',
                details: err.message
            });
        }
    });
}


// --- (REVISED) Analyze API (File Upload) ---
app.post('/analyze', upload.single('dataset'), (req, res) => {
    console.log(`[${new Date().toISOString()}] Received /analyze (upload)`);

    if (!req.file) {
        console.error('-> No file uploaded.');
        return res.status(400).json({ error: 'No dataset file uploaded.' });
    }
    
    // Run analysis and tell it to delete the file after
    runPythonAnalysis(req.file.path, res, true);
});

// --- (NEW) Analyze API (Pre-loaded) ---
app.post('/analyze-preloaded', (req, res) => {
    console.log(`[${new Date().toISOString()}] Received /analyze-preloaded`);
    const { datasetName } = req.body;

    if (!datasetName) {
        return res.status(400).json({ error: 'No dataset name provided.' });
    }

    // Basic security check to prevent directory traversal
    if (datasetName.includes('..') || datasetName.includes('/')) {
         return res.status(400).json({ error: 'Invalid dataset name.' });
    }

    const filePath = path.join(preloadedDir, datasetName);
    
    // Run analysis, DO NOT delete the file (deleteFileAfter = false)
    runPythonAnalysis(filePath, res, false);
});

// --- Start Server ---
app.listen(PORT, () => console.log(`🚀 Server running locally at http://localhost:${PORT}`));