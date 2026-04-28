import { useState, useEffect } from 'react'

interface PolicyPanelProps {
  platform: 'Instagram' | 'Snapchat'
  onPlatformChange: (platform: string) => void
}

interface PlatformPolicy {
  platform: string
  threshold: number
  base_rate: number
  fp_penalty_weight: number
  primary_enforcement_signal: string
  fn_cost_signal: string
  fp_cost_signal: string
  confidence: number
  compiled_at: string
}

export default function PolicyPanel({ platform, onPlatformChange }: PolicyPanelProps) {
  const [isCompiling, setIsCompiling] = useState(false)
  const [progress, setProgress] = useState<string[]>([])
  const [policy, setPolicy] = useState<PlatformPolicy | null>(null)

  // Load cached policy on mount and platform change
  useEffect(() => {
    fetch(`/api/policy/${platform}`)
      .then(res => res.json())
      .then(data => setPolicy(data.policy))
      .catch(err => console.error('Failed to load policy:', err))
  }, [platform])

  const compilePolicy = () => {
    setIsCompiling(true)
    setProgress([])

    // WebSocket for streaming progress
    const ws = new WebSocket(`ws://localhost:8000/ws/compile_policy/${platform}`)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'progress') {
        setProgress(prev => [...prev, data.message])
      } else if (data.type === 'complete') {
        setPolicy(data.policy)
        setIsCompiling(false)
        ws.close()
      } else if (data.type === 'error') {
        console.error('Policy compilation error:', data.message)
        setIsCompiling(false)
        ws.close()
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setIsCompiling(false)
    }
  }

  return (
    <div className="panel">
      <h2>Platform Policy Compiler</h2>

      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem', color: '#cbd5e1' }}>
          Platform:
        </label>
        <select
          value={platform}
          onChange={(e) => onPlatformChange(e.target.value)}
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

      <button
        onClick={compilePolicy}
        disabled={isCompiling}
        className="btn btn-primary"
        style={{ width: '100%', marginBottom: '1rem' }}
      >
        {isCompiling ? 'Compiling...' : 'Compile Policy'}
      </button>

      {isCompiling && progress.length > 0 && (
        <div className="progress-log">
          {progress.map((msg, i) => (
            <div key={i} className="progress-item">{msg}</div>
          ))}
        </div>
      )}

      {policy && !isCompiling && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#0f172a', borderRadius: '0.375rem' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: '#f1f5f9' }}>
            {policy.platform} Policy
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.875rem' }}>
            <div>
              <div style={{ color: '#94a3b8' }}>Threshold (θ*):</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#3b82f6' }}>
                {policy.threshold.toFixed(3)}
                {policy.threshold < 0.2 && <span style={{ marginLeft: '0.5rem', fontSize: '0.875rem', color: '#ef4444' }}>STRICT</span>}
                {policy.threshold > 0.6 && <span style={{ marginLeft: '0.5rem', fontSize: '0.875rem', color: '#4ade80' }}>LENIENT</span>}
              </div>
            </div>

            <div>
              <div style={{ color: '#94a3b8' }}>Base Rate (π):</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 'semibold' }}>
                {(policy.base_rate * 100).toFixed(2)}%
              </div>
            </div>

            <div>
              <div style={{ color: '#94a3b8' }}>FP Penalty:</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 'semibold' }}>
                {policy.fp_penalty_weight.toFixed(2)}x
                {policy.fp_penalty_weight > 0.05 && <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: '#fbbf24' }}>HIGH</span>}
                {policy.fp_penalty_weight <= 0.05 && <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: '#4ade80' }}>LOW</span>}
              </div>
            </div>

            <div>
              <div style={{ color: '#94a3b8' }}>Primary Signal:</div>
              <div style={{ fontSize: '1.125rem', fontWeight: 'semibold' }}>
                {policy.primary_enforcement_signal}
              </div>
            </div>

            <div>
              <div style={{ color: '#94a3b8' }}>FN Cost:</div>
              <div style={{ fontSize: '1.125rem' }}>{policy.fn_cost_signal}</div>
            </div>

            <div>
              <div style={{ color: '#94a3b8' }}>FP Cost:</div>
              <div style={{ fontSize: '1.125rem' }}>{policy.fp_cost_signal}</div>
            </div>

            <div>
              <div style={{ color: '#94a3b8' }}>Confidence:</div>
              <div style={{ fontSize: '1.125rem' }}>{(policy.confidence * 100).toFixed(0)}%</div>
            </div>

            <div>
              <div style={{ color: '#94a3b8' }}>Compiled:</div>
              <div style={{ fontSize: '0.75rem' }}>
                {new Date(policy.compiled_at).toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
