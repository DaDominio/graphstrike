import { useState } from 'react'

interface ControlPanelProps {
  onStart: (platform: string, task: string, seed: number) => void
  onStop: () => void
  onRunTraining: (episodes: number) => void
  isRunning: boolean
}

export default function ControlPanel({ onStart, onStop, onRunTraining, isRunning }: ControlPanelProps) {
  const [platform, setPlatform] = useState('Instagram')
  const [task, setTask] = useState('easy')
  const [seed, setSeed] = useState(0)
  const [numEpisodes, setNumEpisodes] = useState(50)

  return (
    <div className="panel" style={{ marginTop: '1.5rem', maxWidth: '800px', margin: '1.5rem auto 0' }}>
      <h2>Control Panel</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Episode Control */}
        <div>
          <h3 style={{ fontSize: '1.125rem', marginBottom: '1rem', color: '#f1f5f9' }}>
            Single Episode
          </h3>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem' }}>
              Platform:
            </label>
            <select
              value={platform}
              onChange={(e) => setPlatform(e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '0.375rem',
                background: '#0f172a',
                border: '1px solid #475569',
                color: '#e2e8f0',
              }}
            >
              <option>Instagram</option>
              <option>Snapchat</option>
            </select>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem' }}>
              Task:
            </label>
            <select
              value={task}
              onChange={(e) => setTask(e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '0.375rem',
                background: '#0f172a',
                border: '1px solid #475569',
                color: '#e2e8f0',
              }}
            >
              <option>easy</option>
              <option>medium</option>
              <option>hard</option>
            </select>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem' }}>
              Seed:
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(+e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '0.375rem',
                background: '#0f172a',
                border: '1px solid #475569',
                color: '#e2e8f0',
              }}
            />
          </div>

          {!isRunning ? (
            <button
              className="btn btn-primary"
              onClick={() => onStart(platform, task, seed)}
              style={{ width: '100%' }}
            >
              Start Episode
            </button>
          ) : (
            <button
              className="btn"
              onClick={onStop}
              style={{ width: '100%', background: '#ef4444', color: 'white' }}
            >
              Stop Episode
            </button>
          )}
        </div>

        {/* Training Control */}
        <div>
          <h3 style={{ fontSize: '1.125rem', marginBottom: '1rem', color: '#f1f5f9' }}>
            Batch Training
          </h3>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#cbd5e1', fontSize: '0.875rem' }}>
              Number of Episodes:
            </label>
            <input
              type="number"
              value={numEpisodes}
              onChange={(e) => setNumEpisodes(+e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '0.375rem',
                background: '#0f172a',
                border: '1px solid #475569',
                color: '#e2e8f0',
              }}
            />
          </div>

          <div style={{ padding: '1rem', background: '#0f172a', borderRadius: '0.375rem', marginBottom: '1rem', fontSize: '0.875rem' }}>
            <div style={{ color: '#94a3b8', marginBottom: '0.5rem' }}>
              Will train on both platforms:
            </div>
            <div style={{ color: '#cbd5e1' }}>
              • Instagram: {Math.floor(numEpisodes / 2)} episodes
            </div>
            <div style={{ color: '#cbd5e1' }}>
              • Snapchat: {Math.ceil(numEpisodes / 2)} episodes
            </div>
          </div>

          <button
            className="btn btn-secondary"
            onClick={() => onRunTraining(numEpisodes)}
            disabled={isRunning}
            style={{ width: '100%' }}
          >
            Run Training ({numEpisodes} episodes)
          </button>
        </div>
      </div>
    </div>
  )
}
