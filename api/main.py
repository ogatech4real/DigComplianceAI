from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import BytesIO
from services.screening_service import screen_dataframe

app = FastAPI(title="Digital Trade Compliance AI")


def _read_upload(upload: UploadFile) -> pd.DataFrame:
    content = upload.file.read()
    name = upload.filename.lower()
    if name.endswith('.csv'):
        return pd.read_csv(BytesIO(content))
    if name.endswith('.xlsx'):
        return pd.read_excel(BytesIO(content))
    if name.endswith('.json'):
        return pd.read_json(BytesIO(content))
    raise ValueError('Unsupported file type')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/screen-records')
def screen_records(file: UploadFile = File(...)):
    df = _read_upload(file)
    result = screen_dataframe(df)
    if result['status'] != 'ok':
        return result
    return {
        'status': 'ok',
        'rows': len(result['data']),
        'high_risk': int((result['data']['hybrid_risk'] == 'high').sum())
    }
