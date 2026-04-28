# Fake Gang Detection Dashboard

Interactive visualization dashboard for Round 2 platform-adaptive detection.

**Tech Stack**: React 18 + TypeScript + D3.js + FastAPI + WebSockets

---

## Setup

### Backend (FastAPI)

```bash
cd dashboard/backend
pip install -r requirements.txt
python main.py
```

Backend runs on `http://localhost:8000`

API docs: `http://localhost:8000/docs`

### Frontend (React + Vite)

```bash
cd dashboard/frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`

---

## Features

### 1. Policy Compiler Panel
- Select platform (Instagram/Snapchat)
- Click "Compile Policy" to run Bayesian threshold calculation
- Streaming progress via WebSocket
- Policy summary card with:
  - Threshold (θ*) with STRICT/LENIENT badge
  - Base rate (π)
  - FP penalty weight with HIGH/LOW badge
  - Primary enforcement signal
  - FN/FP cost signals
  - Confidence score
  - Compilation timestamp

### 2. Network Graph Panel
- D3.js force-directed layout
- Node colors:
  - 🔴 Red: Gang members (confirmed fakes)
  - 🟢 Green: Real accounts
  - 🟣 Purple: Celebrities (high hub legitimacy)
- Edge styles:
  - Solid: Follows relationship
  - Dashed: Mutual follows
- Interactive:
  - Drag nodes to reposition
  - Zoom and pan
  - Hover for tooltips (account ID, risk, hub score)
- Auto-refreshes every 2 seconds during episode

### 3. Training Panel
- Platform comparison (Instagram vs Snapchat)
- Metrics:
  - Episode count
  - Average score
  - Average precision
  - Average recall
- Recent episodes list with:
  - Episode number and platform
  - Score (color-coded: green >0.85, yellow <0.85)
  - TP/FP/FN breakdown

### 4. Control Panel
- **Single Episode**:
  - Platform selector
  - Task difficulty (easy/medium/hard)
  - Seed input
  - Start/Stop buttons
- **Batch Training**:
  - Episode count input
  - Auto-split between Instagram/Snapchat
  - Run training button

---

## API Endpoints

### WebSocket

- `WS /ws/compile_policy/{platform}` - Policy compilation progress
- `WS /ws/episode/{episode_id}` - Episode state updates
- `WS /ws/training/{training_id}` - Training progress

### REST

- `POST /api/compile_policy` - Compile policy (non-streaming)
- `GET /api/policy/{platform}` - Get cached policy
- `POST /api/episode/start` - Start new episode
- `POST /api/episode/{episode_id}/step` - Execute action
- `GET /api/episode/{episode_id}/state` - Get episode state
- `POST /api/training/start` - Start training loop
- `GET /api/training/{training_id}/metrics` - Get training metrics
- `GET /health` - Health check

---

## Development

### Backend Development

```bash
cd dashboard/backend

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Test policy endpoint
curl http://localhost:8000/api/policy/Instagram

# Test episode start
curl -X POST http://localhost:8000/api/episode/start \
  -H "Content-Type: application/json" \
  -d '{"platform": "Instagram", "task": "easy", "seed": 0}'
```

### Frontend Development

```bash
cd dashboard/frontend

# Install dependencies
npm install

# Run dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Adding New Components

1. Create component file in `src/components/`
2. Import in `App.tsx`
3. Add to panels grid or control panel

Example:
```typescript
// src/components/MetricsChart.tsx
import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

export default function MetricsChart() {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    // D3.js rendering logic
  }, [])

  return <svg ref={svgRef} />
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                           │
│  ┌───────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ PolicyPanel   │  │ NetworkGraph   │  │ TrainingPanel  │ │
│  │ (streaming)   │  │ (D3.js force)  │  │ (metrics)      │ │
│  └───────────────┘  └────────────────┘  └────────────────┘ │
│            │                 │                  │            │
│            └─────────────────┴──────────────────┘            │
│                              │                                │
│                    WebSocket + REST API                       │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────┴────────────────────────────────┐
│                    FastAPI Backend                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Policy Compiler│  │   Environment  │  │  Episode State │ │
│  │ (WebSocket)    │  │  (OpenEnv)     │  │  Management    │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### Data Flow

**Policy Compilation**:
1. User selects platform and clicks "Compile Policy"
2. Frontend opens WebSocket to `/ws/compile_policy/{platform}`
3. Backend calls `policy_compiler.get_policy(platform)`
4. Progress streamed via WebSocket: `{type: "progress", message: "..."}`
5. Complete policy sent: `{type: "complete", policy: {...}}`
6. Frontend renders PolicyCard with metrics

**Episode Execution**:
1. User sets parameters and clicks "Start Episode"
2. Frontend POSTs to `/api/episode/start`
3. Backend creates `FakeGangEnvironment`, calls `reset()`
4. Returns `{episode_id, platform, observation, policy}`
5. Frontend polls `/api/episode/{episode_id}/state` every 2s
6. Returns `{observation, graph_data, metrics}`
7. NetworkGraph updates with new nodes/edges

**Training Loop** (TODO: integrate with agent/train.py):
1. User sets episode count and clicks "Run Training"
2. Frontend POSTs to `/api/training/start`
3. Backend starts training loop in background
4. For each episode, sends WebSocket update: `{type: "episode_complete", ...}`
5. Frontend updates TrainingPanel metrics in real-time

---

## Next Steps

### Phase 1: Core Functionality ✅
- [x] PolicyPanel with WebSocket streaming
- [x] NetworkGraphPanel with D3.js force layout
- [x] TrainingPanel with metrics display
- [x] ControlPanel for episode/training control
- [x] FastAPI backend with REST + WebSocket

### Phase 2: Integration 📋
- [ ] Connect training loop to `agent/train.py`
- [ ] Implement real-time episode stepping (manual mode)
- [ ] Add WebSocket reconnection logic
- [ ] Store training metrics in backend state

### Phase 3: Enhancements 📋
- [ ] Training curve line chart (D3.js)
- [ ] Tool usage breakdown bar chart
- [ ] Precision/Recall scatter plot
- [ ] Export metrics to JSON/CSV
- [ ] Dark/light theme toggle
- [ ] Responsive design for mobile

### Phase 4: Advanced Features 📋
- [ ] Multi-model comparison (Qwen vs Claude vs Llama)
- [ ] Episode replay (step-by-step visualization)
- [ ] Gang registry visualization
- [ ] Custom policy editor
- [ ] A/B testing framework

---

## Troubleshooting

**Backend fails to start**:
- Check Python version (3.10+)
- Install requirements: `pip install -r requirements.txt`
- Check port 8000 is available: `lsof -i :8000`

**Frontend fails to build**:
- Check Node version (18+)
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Clear cache: `rm -rf .vite`

**WebSocket connection fails**:
- Check backend is running on port 8000
- Check CORS settings in `main.py`
- Open browser console for errors

**Graph not rendering**:
- Check D3.js version: `npm list d3`
- Check browser console for errors
- Verify `graphData` is populated: `console.log(graphData)`

**Policy not loading**:
- Check `policy_cache/` directory exists
- Verify Tavily/Groq API keys (or use fallback)
- Check backend logs for compilation errors

---

## Production Deployment

### Backend

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn (production WSGI server)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### Frontend

```bash
# Build production bundle
npm run build

# Serve with nginx or host on Vercel/Netlify
# Output in dist/ directory
```

### Docker (TODO)

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

**Dashboard Status**: ✅ Core implementation complete, ready for integration testing

**Next**: Connect training loop and test end-to-end flow
