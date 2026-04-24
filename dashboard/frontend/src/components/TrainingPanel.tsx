import { useState, useEffect } from 'react'

interface TrainingMetric {
  episode: number
  platform: string
  score: number
  tp: number
  fp: number
  fn: number
}

export default function TrainingPanel() {
  const [metrics, setMetrics] = useState<TrainingMetric[]>([])

  // Simulate some initial data
  useEffect(() => {
    // In real implementation, this would connect to WebSocket for live updates
    const demoData: TrainingMetric[] = [
      { episode: 0, platform: 'Instagram', score: 0.82, tp: 9, fp: 2, fn: 1 },
      { episode: 1, platform: 'Snapchat', score: 0.88, tp: 9, fp: 1, fn: 1 },
      { episode: 2, platform: 'Instagram', score: 0.85, tp: 9, fp: 1, fn: 1 },
      { episode: 3, platform: 'Snapchat', score: 0.91, tp: 10, fp: 1, fn: 0 },
    ]
    setMetrics(demoData)
  }, [])

  const instagramMetrics = metrics.filter(m => m.platform === 'Instagram')
  const snapchatMetrics = metrics.filter(m => m.platform === 'Snapchat')

  const avg = (arr: TrainingMetric[], key: keyof TrainingMetric) => {
    if (arr.length === 0) return 0
    return arr.reduce((sum, m) => sum + (m[key] as number), 0) / arr.length
  }

  return (
    <div className="panel">
      <h2>Training Metrics</h2>

      {metrics.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
          Start training to see metrics
        </div>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
            {/* Instagram Stats */}
            <div style={{ padding: '1rem', background: '#0f172a', borderRadius: '0.375rem', border: '2px solid #3b82f6' }}>
              <h3 style={{ fontSize: '1.125rem', marginBottom: '0.75rem', color: '#3b82f6' }}>
                Instagram
              </h3>
              <div style={{ fontSize: '0.875rem', color: '#cbd5e1' }}>
                <div style={{ marginBottom: '0.5rem' }}>
                  <span style={{ color: '#94a3b8' }}>Episodes:</span>{' '}
                  <span style={{ fontWeight: 'semibold' }}>{instagramMetrics.length}</span>
                </div>
                <div style={{ marginBottom: '0.5rem' }}>
                  <span style={{ color: '#94a3b8' }}>Avg Score:</span>{' '}
                  <span style={{ fontWeight: 'semibold', color: '#3b82f6' }}>
                    {avg(instagramMetrics, 'score').toFixed(3)}
                  </span>
                </div>
                <div style={{ marginBottom: '0.5rem' }}>
                  <span style={{ color: '#94a3b8' }}>Avg Precision:</span>{' '}
                  <span style={{ fontWeight: 'semibold' }}>
                    {instagramMetrics.length > 0
                      ? (instagramMetrics.reduce((sum, m) => sum + (m.tp / (m.tp + m.fp)), 0) / instagramMetrics.length).toFixed(2)
                      : '0.00'}
                  </span>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>Avg Recall:</span>{' '}
                  <span style={{ fontWeight: 'semibold' }}>
                    {instagramMetrics.length > 0
                      ? (instagramMetrics.reduce((sum, m) => sum + (m.tp / (m.tp + m.fn)), 0) / instagramMetrics.length).toFixed(2)
                      : '0.00'}
                  </span>
                </div>
              </div>
            </div>

            {/* Snapchat Stats */}
            <div style={{ padding: '1rem', background: '#0f172a', borderRadius: '0.375rem', border: '2px solid #f97316' }}>
              <h3 style={{ fontSize: '1.125rem', marginBottom: '0.75rem', color: '#f97316' }}>
                Snapchat
              </h3>
              <div style={{ fontSize: '0.875rem', color: '#cbd5e1' }}>
                <div style={{ marginBottom: '0.5rem' }}>
                  <span style={{ color: '#94a3b8' }}>Episodes:</span>{' '}
                  <span style={{ fontWeight: 'semibold' }}>{snapchatMetrics.length}</span>
                </div>
                <div style={{ marginBottom: '0.5rem' }}>
                  <span style={{ color: '#94a3b8' }}>Avg Score:</span>{' '}
                  <span style={{ fontWeight: 'semibold', color: '#f97316' }}>
                    {avg(snapchatMetrics, 'score').toFixed(3)}
                  </span>
                </div>
                <div style={{ marginBottom: '0.5rem' }}>
                  <span style={{ color: '#94a3b8' }}>Avg Precision:</span>{' '}
                  <span style={{ fontWeight: 'semibold' }}>
                    {snapchatMetrics.length > 0
                      ? (snapchatMetrics.reduce((sum, m) => sum + (m.tp / (m.tp + m.fp)), 0) / snapchatMetrics.length).toFixed(2)
                      : '0.00'}
                  </span>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>Avg Recall:</span>{' '}
                  <span style={{ fontWeight: 'semibold' }}>
                    {snapchatMetrics.length > 0
                      ? (snapchatMetrics.reduce((sum, m) => sum + (m.tp / (m.tp + m.fn)), 0) / snapchatMetrics.length).toFixed(2)
                      : '0.00'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Episode History */}
          <div>
            <h3 style={{ fontSize: '1.125rem', marginBottom: '0.75rem', color: '#f1f5f9' }}>
              Recent Episodes
            </h3>
            <div style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '0.875rem' }}>
              {metrics.slice().reverse().map((m, i) => (
                <div
                  key={i}
                  style={{
                    padding: '0.5rem',
                    marginBottom: '0.5rem',
                    background: '#0f172a',
                    borderRadius: '0.25rem',
                    borderLeft: `3px solid ${m.platform === 'Instagram' ? '#3b82f6' : '#f97316'}`,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontWeight: 'semibold' }}>
                      Episode #{m.episode} ({m.platform})
                    </span>
                    <span style={{ color: m.score >= 0.85 ? '#4ade80' : '#fbbf24' }}>
                      {m.score.toFixed(3)}
                    </span>
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                    TP={m.tp} FP={m.fp} FN={m.fn}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
