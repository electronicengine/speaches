param([string]$SshTarget = 'verasist-server')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$composeArgs = @('compose', '-f', (Join-Path $projectRoot 'compose.cuda.yaml'), '-f', (Join-Path $projectRoot 'compose.local.yaml'))
$logs = & docker @composeArgs logs --no-color --tail 150 tunnel 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Cannot read tunnel logs' }
$urls = [regex]::Matches(($logs -join "`n"), 'https://[a-z0-9-]+\.trycloudflare\.com')
if ($urls.Count -eq 0) { throw 'Tunnel URL is not ready' }
$keyLine = Get-Content (Join-Path $projectRoot '.env') | Where-Object { $_ -match '^API_KEY=.' } | Select-Object -First 1
if (-not $keyLine) { throw 'API_KEY is missing' }
$settings = @{ base_url = $urls[$urls.Count - 1].Value + '/v1'; api_key = $keyLine.Substring(8) } | ConvertTo-Json -Compress
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($settings))
$python = @'
import base64, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

payload = '__PAYLOAD__'

def run_app(script):
    result = subprocess.run(
        ['docker', 'compose', 'exec', '-T', 'api', 'python', '-W', 'ignore', '-'],
        input=script, text=True, capture_output=True, cwd='/root/verasist',
    )
    if result.returncode:
        # Do not echo raw exception messages: they can contain credentials.
        raise RuntimeError('Verasist configuration operation failed; credentials were not printed')
    return result.stdout.strip()

snapshot = run_app('''
import asyncio, json
from api.db import db_client
async def main():
    saved = []
    for service in ('stt', 'tts'):
        row = await db_client.get_platform_provider_credential(service, 'speaches')
        saved.append({'service_type': service, 'provider': 'speaches',
            'credentials': row.credentials if row else None,
            'is_active': row.is_active if row else False,
            'created_by': row.created_by if row else None})
    print(json.dumps(saved))
asyncio.run(main())
''')
saved = json.loads(snapshot)
folder = Path('/root/.config/verasist/local-speaches')
folder.mkdir(mode=0o700, parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
backup = folder / (stamp + '-credentials.json')
with os.fdopen(os.open(str(backup), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
    json.dump(saved, f)
print('Previous STT/TTS credentials backed up:', backup)

app_script = '''
import asyncio, base64, json, httpx
from api.db import db_client
from api.services.configuration.platform_credentials import upsert_platform_credential, apply_platform_credentials
from api.services.configuration.local_models import list_local_models
from api.services.configuration.registry import SpeachesSTTConfiguration, SpeachesTTSConfiguration
from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration

settings = json.loads(base64.b64decode('__SETTINGS__'))
old = json.loads(base64.b64decode('__OLD__'))

async def main():
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(settings['base_url'] + '/models', headers={'Authorization': 'Bearer ' + settings['api_key']})
        r.raise_for_status()
        ids = {m['id'] for m in r.json()['data']}
        assert {'mihuai/turkish-stt', 'canberkkkkkk/ema-lightning'} <= ids
    try:
        for previous in old:
            credentials = dict(previous['credentials'] or {})
            credentials.update(settings)
            await upsert_platform_credential(previous['service_type'], 'speaches', credentials,
                created_by=previous['created_by'], is_active=True)
        for service in ('stt', 'tts'):
            result = await list_local_models(service)
            assert result.status == 'ready' and result.models, service
            print(service.upper() + ' picker:', json.dumps(result.model_dump()))
        cfg = EffectiveAIModelConfiguration(
            stt=SpeachesSTTConfiguration(model='mihuai/turkish-stt', language='tr'),
            tts=SpeachesTTSConfiguration(model='canberkkkkkk/ema-lightning', voice='default'))
        resolved = await apply_platform_credentials(cfg, sections=['stt','tts'])
        for service in ('stt', 'tts'):
            section = getattr(resolved, service)
            assert section.base_url == settings['base_url']
            assert section.api_key == settings['api_key']
            print(service.upper() + ' runtime endpoint:', section.base_url)
        print('Model package discovery and runtime credentials: OK')
    except BaseException:
        for previous in old:
            if previous['credentials'] is None:
                await db_client.delete_platform_provider_credential(previous['service_type'], 'speaches')
            else:
                await upsert_platform_credential(previous['service_type'], 'speaches', previous['credentials'],
                    created_by=previous['created_by'], is_active=previous['is_active'])
        raise

asyncio.run(main())
'''
app_script = app_script.replace('__SETTINGS__', payload).replace('__OLD__', base64.b64encode(snapshot.encode()).decode())
print(run_app(app_script))
'@
$python = $python.Replace('__PAYLOAD__', $payload)
$python | & ssh -o BatchMode=yes -o ConnectTimeout=8 $SshTarget python3 -
if ($LASTEXITCODE -ne 0) { throw 'Verasist connection failed' }
