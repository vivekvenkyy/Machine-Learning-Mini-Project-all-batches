const fs = require('fs');
const https = require('https');
const path = require('path');

function readEnvFile(envPath) {
  try {
    const content = fs.readFileSync(envPath, 'utf8');
    const match = content.match(/^VITE_GEMINI_API_KEY=(.*)$/m);
    if (match) return match[1].trim();
  } catch (e) {
    // ignore
  }
  return null;
}

const envPath = path.join(process.cwd(), '.env');
const apiKey = readEnvFile(envPath) || process.env.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY;

if (!apiKey) {
  console.error('No API key found. Put VITE_GEMINI_API_KEY=your_key in .env or set VITE_GEMINI_API_KEY env var.');
  process.exit(1);
}

const url = `https://generativelanguage.googleapis.com/v1/models?key=${apiKey}`;

https.get(url, (res) => {
  const { statusCode } = res;
  let rawData = '';
  res.on('data', (chunk) => { rawData += chunk; });
  res.on('end', () => {
    console.log('HTTP status:', statusCode);
    try {
      const parsed = JSON.parse(rawData);
      console.log('Full response JSON:\n', JSON.stringify(parsed, null, 2));

      // Attempt to extract model names from common response shapes
      const models = parsed.models || parsed.model || parsed.models || parsed;
      if (Array.isArray(parsed.models)) {
        console.log('\nAvailable models:');
        parsed.models.forEach((m) => {
          if (typeof m === 'string') console.log('- ' + m);
          else if (m.name) console.log('- ' + m.name);
          else console.log('- ' + JSON.stringify(m));
        });
      } else if (parsed.model) {
        console.log('\nModel info:');
        console.log(JSON.stringify(parsed.model, null, 2));
      } else if (Array.isArray(models)) {
        console.log('\nModels array:');
        models.forEach((m) => console.log('- ' + (m.name || JSON.stringify(m))));
      } else {
        console.log('\nCould not parse a models array from the response. See full JSON above.');
      }
    } catch (e) {
      console.error('Error parsing response JSON:', e.message);
      console.log('Raw response:\n', rawData);
    }
  });
}).on('error', (e) => {
  console.error('Request error:', e.message);
});
