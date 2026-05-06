from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import uvicorn

if __name__ == '__main__':
    uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=True)
