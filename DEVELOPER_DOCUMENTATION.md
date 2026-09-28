# PrintOpt AI --- Developer Documentation

**Project:** PrintOpt AI --- AI-Driven Additive Manufacturing Process
Optimizer\
**Process:** Laser Powder Bed Fusion (LPBF) of Ti-6Al-4V\
**Developers:** Hardik Sonu, Samreen Utta Ur Rehman\
**Supervisor:** Dr. Ing. Waseem Amin\
**Institution:** University of the Punjab, Lahore\
**Version:** 1.0 --- 2026

## 1. Purpose

This document is the developer reference for maintaining, running,
debugging, and extending PrintOpt AI. It records the actual project
locations, frontend/backend structure, API endpoints, data/model
locations, configuration, development commands, and important
implementation rules.

## 2. Project Root

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI
```

Main applications:

``` text
PrintOpt AI/
├── frontend/
│   └── printopt-ai-frontend/
└── ML Model/
```

## 3. Frontend

Exact location:

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend
```

Technology:

-   Next.js
-   React
-   TypeScript
-   Tailwind CSS
-   Recharts
-   Lucide React

Run:

``` powershell
cd "C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend"
npm run dev
```

URL:

``` text
http://localhost:3000
```

### Important frontend files

``` text
app\layout.tsx
app\globals.css
app\page.tsx
services\api.ts
types\index.ts
lib\ved.ts
components\layout\Sidebar.tsx
components\layout\PageHeader.tsx
components\ui\Card.tsx
components\ui\MetricCard.tsx
```

### Application routes

``` text
/                  Overview
/dataset           Dataset
/prediction        Prediction
/optimization      Optimization
/process-map       Process Map
/model             Model
/sources           Sources
/privacy-policy    Privacy Policy
```

## 4. Backend / ML

Exact location:

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model
```

Technology:

-   Python
-   FastAPI
-   Uvicorn
-   scikit-learn
-   Joblib
-   CSV-based data pipeline

Run:

``` powershell
cd "C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model"
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

Backend URL:

``` text
http://127.0.0.1:8000
```

Main backend file:

``` text
ML Model\api\main.py
```

## 5. Environment Configuration

Frontend environment file:

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend\.env.local
```

Current value:

``` env
NEXT_PUBLIC_API_URL=https://printopt-ai-production.up.railway.app
```

The frontend API service reads this variable instead of hardcoding the
backend URL throughout the application.

## 6. API Service

Frontend API service:

``` text
frontend\printopt-ai-frontend\services\api.ts
```

Current services:

``` text
datasetApi
modelApi
predictionApi
optimizationApi
processMapApi
sourcesApi
```

Type definitions:

``` text
frontend\printopt-ai-frontend\types\index.ts
```

When an API response changes, update the corresponding TypeScript
interface as well as the API service.

## 7. Backend API

  --------------------------------------------------------------------------
  Method                  Endpoint                   Purpose
  ----------------------- -------------------------- -----------------------
  GET                     `/`                        API root/status

  GET                     `/health`                  Backend health

  POST                    `/predict`                 Predict UTS, YS, and
                                                     elongation

  GET                     `/api/dataset`             Return processed
                                                     experimental records

  GET                     `/api/process-map`         Return VED/property
                                                     process-map data

  POST                    `/api/optimize`            Constrained process
                                                     optimization

  GET                     `/api/model/status`        Model status

  GET                     `/api/model/metrics`       Evaluation metrics

  GET                     `/api/model/diagnostics`   Feature importance and
                                                     held-out predictions

  GET                     `/api/sources`             Source groups and
                                                     record counts
  --------------------------------------------------------------------------

## 8. Dataset

Primary processed dataset:

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model\data\processed\ti64_lpbf_processed.csv
```

Current dataset size:

``` text
173 records
```

Important columns:

``` text
Experiment_ID
Source_ID
Powder_Size_um
Laser_Spot_um
Laser_Power_W
Scanning_Speed_mm_s
Hatch_Distance_um
Layer_Thickness_um
Hatch_Distance_mm
Layer_Thickness_mm
VED_J_mm3
UTS_MPa
YS_MPa
Elongation_pct
Reference
```

Other processed files are located in:

``` text
ML Model\data\processed\
```

including:

``` text
test.csv
test_v2.csv
ti64_lpbf_full_sensitivity.csv
ti64_lpbf_training_candidate.csv
ti64_lpbf_verified.csv
train.csv
train_v2.csv
```

Do not replace the active processed dataset without checking column
names, units, missing values, source IDs, features, and targets.

## 9. Active ML Model

Active model:

``` text
Random Forest V2
```

Expected file:

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model\models\random_forest_v2.joblib
```

Model features:

``` text
Powder_Size_um
Laser_Spot_um
Laser_Power_W
Scanning_Speed_mm_s
Hatch_Distance_um
Layer_Thickness_um
VED_J_mm3
```

Targets:

``` text
UTS_MPa
YS_MPa
Elongation_pct
```

## 10. VED

The engineering feature used by the application is volumetric energy
density:

``` text
VED = P / (v × h × t)
```

where:

``` text
P = laser power
v = scanning speed
h = hatch distance
t = layer thickness
```

Hatch distance and layer thickness must be converted to millimetres
before the calculation.

Frontend VED logic:

``` text
frontend\printopt-ai-frontend\lib\ved.ts
```

Backend VED logic:

``` text
ML Model\api\main.py
```

## 11. Optimization

Optimization is currently implemented in:

``` text
ML Model\api\main.py
```

Current search size:

``` text
10,000 candidate evaluations
```

Optimization variables:

``` text
Laser Power
Scanning Speed
Hatch Distance
Layer Thickness
```

Current fixed defaults:

``` text
Powder size = 35 µm
Laser spot = 81 µm
```

Supported objectives:

``` text
maximize_uts
maximize_yield_strength
maximize_elongation
multi_objective
```

Optimization output is model-derived and must be experimentally
validated before manufacturing use.

## 12. Evaluation Files

Evaluation directory:

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model\evaluation
```

Important files:

``` text
baseline_results_v2.csv
baseline_feature_importance_v2.csv
test_predictions.csv
```

Current evaluation contains 28 source-aware held-out samples.

Important developer rule:

> Never replace real evaluation results with hardcoded performance
> claims.

The current Random Forest V2 R² values are negative for UTS, Yield
Strength, and Elongation. The application should therefore remain
described as an academic research prototype, not as a validated
industrial predictive system.

## 13. Source Provenance

Source information is derived from:

``` text
ML Model\data\processed\ti64_lpbf_processed.csv
```

Relevant fields:

``` text
Source_ID
Reference
```

The `/api/sources` endpoint groups records by `Source_ID` and returns
record counts.

## 14. Frontend Architecture

General flow:

``` text
User
  ↓
Next.js Page
  ↓
services/api.ts
  ↓
FastAPI
  ↓
Dataset / ML Model / Evaluation Files
  ↓
JSON
  ↓
Frontend UI
```

Prediction flow:

``` text
Parameters
  ↓
Prediction Page
  ↓
predictionApi.predict()
  ↓
POST /predict
  ↓
VED calculation
  ↓
Random Forest V2
  ↓
UTS / YS / Elongation
```

Optimization flow:

``` text
Objective + Constraints
  ↓
Optimization Page
  ↓
POST /api/optimize
  ↓
Candidate Search
  ↓
ML Predictions
  ↓
Recommended Parameters
```

## 15. Adding a New Frontend Page

1.  Create a route under:

``` text
frontend\printopt-ai-frontend\app
```

2.  Add `page.tsx`.
3.  Add the API method to `services/api.ts` if needed.
4.  Add or update interfaces in `types/index.ts`.
5.  Reuse existing layout/UI components.
6.  Add navigation if the page should be globally accessible.
7.  Test loading, success, error, and empty states.

## 16. Adding a New Backend Endpoint

1.  Open:

``` text
ML Model\api\main.py
```

2.  Define request/response models where appropriate.
3.  Validate input.
4.  Add the FastAPI route.
5.  Return structured JSON.
6.  Add the frontend API method.
7.  Add TypeScript types.
8.  Test the endpoint directly.
9.  Test the frontend integration.

## 17. Replacing the ML Model

Before replacing the model:

-   verify feature order
-   verify units
-   verify target order
-   verify preprocessing
-   verify serialization
-   update the model path
-   update model status
-   update evaluation files
-   test `/health`
-   test `/predict`
-   test model metrics
-   test diagnostics
-   test optimization

Do not replace only the `.joblib` file if its feature schema is
incompatible.

## 18. Common Development Issue

The backend must be launched from:

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model
```

Correct:

``` powershell
cd "C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model"
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

Running `uvicorn api.main:app --reload` from the parent `PrintOpt AI`
directory can produce:

``` text
ModuleNotFoundError: No module named 'api'
```

## 19. Backend Import Requirement

If `HTTPException` is used in `main.py`, the import must be:

``` python
from fastapi import FastAPI, HTTPException
```

## 20. Developer Testing Checklist

### Backend

-   [ ] FastAPI starts
-   [ ] `/health` returns healthy
-   [ ] Model loads
-   [ ] Dataset loads
-   [ ] Prediction works
-   [ ] Optimization works
-   [ ] Metrics endpoint works
-   [ ] Diagnostics endpoint works
-   [ ] Sources endpoint works

### Frontend

-   [ ] Next.js starts
-   [ ] Overview loads
-   [ ] Dataset loads real data
-   [ ] Prediction calls backend
-   [ ] Optimization calls backend
-   [ ] Process Map loads real data
-   [ ] Model diagnostics load
-   [ ] Sources load
-   [ ] Privacy Policy loads
-   [ ] Footer renders
-   [ ] Navigation works
-   [ ] Loading/error states work

## 21. Research Safety

PrintOpt AI is an academic research and engineering prototype.

It should not be described as:

-   certified manufacturing software
-   closed-loop machine control
-   guaranteed optimal parameter generation
-   experimentally validated process certification
-   a replacement for qualified engineering judgment

Recommendations generated by the optimization model require experimental
validation.

## 22. Future Refactoring

As the project grows, the backend can eventually be separated into:

``` text
ML Model/
├── api/
│   ├── main.py
│   ├── routes/
│   └── schemas/
├── services/
│   ├── prediction_service.py
│   ├── optimization_service.py
│   └── dataset_service.py
├── preprocessing/
├── models/
├── data/
├── evaluation/
└── tests/
```

This is a future architecture direction; it is not required for the
current working prototype.

## 23. Complete File Location Reference

### Project root

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI
```

### Frontend

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend
```

### Backend

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model
```

### Main API

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model\api\main.py
```

### Dataset

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model\data\processed\ti64_lpbf_processed.csv
```

### Active model

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model\models\random_forest_v2.joblib
```

### Evaluation

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\ML Model\evaluation
```

### Frontend API service

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend\services\api.ts
```

### Frontend types

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend\types\index.ts
```

### Global CSS

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend\app\globals.css
```

### Root layout

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend\app\layout.tsx
```

### Environment

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\frontend\printopt-ai-frontend\.env.local
```

### Developer documentation

``` text
C:\Users\Ahmadcomputer\.gemini\antigravity\scratch\PrintOpt AI\DEVELOPER_DOCUMENTATION.md
```

## 24. Attribution

**PrintOpt AI**

Developed by:

**Hardik Sonu**\
**Samreen Utta Ur Rehman**

Supervised by:

**Dr. Ing. Waseem Amin**

**University of the Punjab, Lahore**

© 2026 PrintOpt AI. All rights reserved.

------------------------------------------------------------------------

**End of Developer Documentation**
