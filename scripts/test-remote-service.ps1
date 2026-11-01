param(
    [string]$SshTarget = 'verasist-server',
    [string]$SttModel = 'mihuai/turkish-stt'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$composeArgs = @('compose', '-f', (Join-Path $projectRoot 'compose.cuda.yaml'), '-f', (Join-Path $projectRoot 'compose.local.yaml'))
$logs = & docker @composeArgs logs --no-color --tail 150 tunnel 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Cannot read tunnel logs' }
$urls = [regex]::Matches(($logs -join "`n"), 'https://[a-z0-9-]+\.trycloudflare\.com')
if ($urls.Count -eq 0) { throw 'Tunnel URL is not ready' }
$keyLine = Get-Content (Join-Path $projectRoot '.env') | Where-Object { $_ -match '^API_KEY=.' } | Select-Object -First 1
if (-not $keyLine) { throw 'API_KEY is missing from .env' }
$settings = @{ url = $urls[$urls.Count - 1].Value; key = $keyLine.Substring(8); stt_model = $SttModel } | ConvertTo-Json -Compress
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($settings))
$python = @'
import base64, json, urllib.request, urllib.error, uuid, time

settings = json.loads(base64.b64decode('__PAYLOAD__'))
url, key = settings['url'], settings['key']

def request(path, data=None, content_type=None, authenticated=True):
    headers = {'Authorization': 'Bearer ' + key} if authenticated else {}
    if content_type:
        headers['Content-Type'] = content_type
    req = urllib.request.Request(url + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.status, response.read()

try:
    request('/v1/models', authenticated=False)
    raise RuntimeError('Unauthenticated request was accepted')
except urllib.error.HTTPError as exc:
    assert exc.code in (401, 403), exc.code
    print('Authentication: OK (unauthenticated requests rejected)')

status, body = request('/v1/models')
models = json.loads(body)['data']
print('Remote API:', status, url + '/v1')
print('Models:', ', '.join(model['id'] for model in models))
assert any(m['id'] == 'canberkkkkkk/ema-lightning' for m in models)
assert any(m['id'] == settings['stt_model'] for m in models)

started = time.monotonic()
status, audio = request('/v1/audio/speech', json.dumps({
    'model': 'canberkkkkkk/ema-lightning', 'voice': 'default',
    'input': 'Merhaba, Verasist sesli asistan testine hos geldiniz.',
    'response_format': 'wav', 'sample_rate': 16000, 'seed': 0,
}).encode(), 'application/json')
assert audio[:4] == b'RIFF' and len(audio) > 3200
print('TTS: OK,', len(audio), 'bytes,', round(time.monotonic() - started, 2), 'seconds')

boundary = uuid.uuid4().hex
parts = []
for name, value in [('model', settings['stt_model']), ('language', 'tr')]:
    parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name + '"\r\n\r\n' + value + '\r\n').encode())
parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="test.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode() + audio + b'\r\n')
parts.append(('--' + boundary + '--\r\n').encode())
started = time.monotonic()
status, body = request('/v1/audio/transcriptions', b''.join(parts), 'multipart/form-data; boundary=' + boundary)
text = json.loads(body)['text']
assert text.strip(), 'Empty transcription'
print('STT: OK,', repr(text), ',', round(time.monotonic() - started, 2), 'seconds')
'@
$python = $python.Replace('__PAYLOAD__', $payload)
$python | & ssh -o BatchMode=yes -o ConnectTimeout=8 $SshTarget 'cd /root/verasist && docker compose exec -T api python -'
if ($LASTEXITCODE -ne 0) { throw 'Remote verification failed' }
