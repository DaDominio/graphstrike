import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

interface NetworkGraphPanelProps {
  episodeId: string | null
}

interface Node {
  id: string
  fake_risk_score: number
  hub_legitimacy_score: number
  is_fake: boolean
}

interface Edge {
  source: string
  target: string
  is_mutual: boolean
}

interface GraphData {
  nodes: Node[]
  edges: Edge[]
}

export default function NetworkGraphPanel({ episodeId }: NetworkGraphPanelProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [graphData, setGraphData] = useState<GraphData | null>(null)

  // Fetch episode state when episodeId changes
  useEffect(() => {
    if (!episodeId) return

    const fetchState = async () => {
      try {
        const response = await fetch(`/api/episode/${episodeId}/state`)
        const data = await response.json()
        setGraphData(data.graph_data)
      } catch (error) {
        console.error('Failed to fetch episode state:', error)
      }
    }

    fetchState()

    // Poll for updates every 2 seconds
    const interval = setInterval(fetchState, 2000)
    return () => clearInterval(interval)
  }, [episodeId])

  // Render D3 graph
  useEffect(() => {
    if (!svgRef.current || !graphData || graphData.nodes.length === 0) return

    const width = 600
    const height = 400

    // Clear previous
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .style('background', '#0f172a')

    const g = svg.append('g')

    // Zoom behavior
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 5])
        .on('zoom', (event) => {
          g.attr('transform', event.transform)
        })
    )

    // Force simulation
    const simulation = d3.forceSimulation(graphData.nodes as any)
      .force('link', d3.forceLink(graphData.edges)
        .id((d: any) => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30))

    // Draw edges
    const link = g.append('g')
      .selectAll('line')
      .data(graphData.edges)
      .enter().append('line')
      .attr('stroke', '#475569')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => d.is_mutual ? 2 : 1)
      .attr('stroke-dasharray', (d: any) => d.is_mutual ? '5,5' : '')

    // Draw nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(graphData.nodes)
      .enter().append('circle')
      .attr('r', (d: any) => 8 + d.fake_risk_score * 15)
      .attr('fill', (d: any) => {
        if (d.is_fake) return '#ef4444' // Gang members
        if (d.hub_legitimacy_score > 0.7) return '#8b5cf6' // Celebrities
        return '#4ade80' // Real accounts
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .call(d3.drag<SVGCircleElement, any>()
        .on('start', (event: any, d: any) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event: any, d: any) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event: any, d: any) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      )

    // Add tooltips
    node.append('title')
      .text((d: any) => `${d.id}\nRisk: ${d.fake_risk_score.toFixed(3)}\nHub: ${d.hub_legitimacy_score.toFixed(2)}`)

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y)
    })

  }, [graphData])

  return (
    <div className="panel">
      <h2>Social Network Graph</h2>

      {!episodeId && (
        <div style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
          Start an episode to visualize the network
        </div>
      )}

      {episodeId && (
        <>
          <svg ref={svgRef} style={{ width: '100%', borderRadius: '0.375rem' }} />

          <div className="legend">
            <div>
              <span className="dot red"></span>
              Gang Members
            </div>
            <div>
              <span className="dot green"></span>
              Real Accounts
            </div>
            <div>
              <span className="dot purple"></span>
              Celebrities
            </div>
          </div>
        </>
      )}
    </div>
  )
}
