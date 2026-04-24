import { useState } from 'react'
import PolicyPanel from './components/PolicyPanel'
import NetworkGraphPanel from './components/NetworkGraphPanel'
import TrainingPanel from './components/TrainingPanel'
import ControlPanel from './components/ControlPanel'

interface AppState {
  selectedPlatform: 'Instagram' | 'Snapchat'
  episodeSeed: number
  episodeTask: 'easy' | 'medium' | 'hard'
  isRunning: boolean
  episodeId: string | null
}

function App() {
  const [state, setState] = useState<AppState>({
    selectedPlatform: 'Instagram',
    episodeSeed: 0,
    episodeTask: 'easy',
    isRunning: false,
    episodeId: null,
  })

  const handleStartEpisode = async (platform: string, task: string, seed: number) => {
    try {
      const response = await fetch('/api/episode/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ platform, task, seed }),
      })

      const data = await response.json()

      setState(prev => ({
        ...prev,
        isRunning: true,
        episodeId: data.episode_id,
        selectedPlatform: data.platform,
      }))
    } catch (error) {
      console.error('Failed to start episode:', error)
    }
  }

  const handleStopEpisode = () => {
    setState(prev => ({
      ...prev,
      isRunning: false,
      episodeId: null,
    }))
  }

  const handleRunTraining = async (episodes: number) => {
    try {
      const response = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episodes, task: state.episodeTask }),
      })

      const data = await response.json()
      console.log('Training started:', data.training_id)
    } catch (error) {
      console.error('Failed to start training:', error)
    }
  }

  return (
    <div className="dashboard-container">
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', color: '#f1f5f9' }}>
          🕵️ Fake Gang Detection Dashboard
        </h1>
        <p style={{ color: '#94a3b8', marginTop: '0.5rem' }}>
          Platform-Adaptive Detection with Real-Time Visualization
        </p>
      </header>

      <div className="panels-grid">
        <PolicyPanel
          platform={state.selectedPlatform}
          onPlatformChange={(p) => setState(prev => ({ ...prev, selectedPlatform: p as any }))}
        />

        <NetworkGraphPanel
          episodeId={state.episodeId}
        />

        <TrainingPanel />
      </div>

      <ControlPanel
        onStart={handleStartEpisode}
        onStop={handleStopEpisode}
        onRunTraining={handleRunTraining}
        isRunning={state.isRunning}
      />
    </div>
  )
}

export default App
