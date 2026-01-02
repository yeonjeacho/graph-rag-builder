import { useEffect, useRef } from 'react';
import { Network, type Options } from 'vis-network';
import { DataSet } from 'vis-data';
import type { GraphNode, GraphRelationship } from '../api';

interface GraphVisualizationProps {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
  height?: string;
}

const labelColors: Record<string, string> = {
  CONCEPT: '#4f46e5',
  Acronym: '#059669',
  TechnicalConcept: '#0891b2',
  Organization: '#dc2626',
  Person: '#ea580c',
  Location: '#7c3aed',
  Technology: '#2563eb',
  Protocol: '#0d9488',
  Standard: '#9333ea',
  Channel: '#65a30d',
  Entity: '#6b7280',
};

function getNodeColor(labels: string[]): string {
  for (const label of labels) {
    if (labelColors[label]) {
      return labelColors[label];
    }
  }
  return '#6b7280';
}

export default function GraphVisualization({ nodes, relationships, height = '400px' }: GraphVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);

  useEffect(() => {
    if (!containerRef.current || nodes.length === 0) return;

    const visNodes = new DataSet(
      nodes.map((node) => ({
        id: node.id,
        label: String(node.properties.name || `Node ${node.id}`),
        title: `${node.labels.join(', ')}\n${JSON.stringify(node.properties, null, 2)}`,
        color: {
          background: getNodeColor(node.labels),
          border: getNodeColor(node.labels),
          highlight: { background: getNodeColor(node.labels), border: '#1f2937' },
        },
        font: { color: '#ffffff', size: 14 },
        shape: 'box' as const,
        borderWidth: 2,
        shadow: true,
      }))
    );

    const visEdges = new DataSet(
      relationships.map((rel, index) => ({
        id: `edge-${index}`,
        from: rel.startNodeId,
        to: rel.endNodeId,
        label: rel.type,
        arrows: 'to' as const,
        color: { color: '#9ca3af', highlight: '#4b5563' },
        font: { size: 11, color: '#6b7280', strokeWidth: 0, background: '#ffffff' },
        smooth: { enabled: true, type: 'curvedCW' as const, roundness: 0.2 },
      }))
    );

    const options: Options = {
      nodes: {
        shape: 'box',
        margin: { top: 10, right: 10, bottom: 10, left: 10 },
        font: { size: 14 },
      },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.8 } },
        smooth: { enabled: true, type: 'curvedCW', roundness: 0.2 },
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -50,
          centralGravity: 0.01,
          springLength: 150,
          springConstant: 0.08,
        },
        stabilization: { iterations: 100 },
      },
      interaction: { hover: true, tooltipDelay: 200, zoomView: true, dragView: true },
      layout: { improvedLayout: true },
    };

    networkRef.current = new Network(containerRef.current, { nodes: visNodes, edges: visEdges }, options);

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, relationships]);

  if (nodes.length === 0) {
    return (
      <div className="flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200" style={{ height }}>
        <p className="text-gray-500">그래프 데이터가 없습니다</p>
      </div>
    );
  }

  return <div ref={containerRef} className="bg-white rounded-lg border border-gray-200" style={{ height }} />;
}
