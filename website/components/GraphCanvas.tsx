'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import UserCard from '@/components/UserCard';
import type { Profile } from '@/types';
import UserPanel from '@/components/UserPanel';

interface Node extends Profile {
  x: number;
  y: number;
  fx?: number;
  fy?: number;
}

interface GraphCanvasProps {
  nodes: Node[];
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
}

const CARD_WIDTH = 300;
const CARD_HEIGHT = 160;

export default function GraphCanvas({ nodes, setNodes }: GraphCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement>(null);
  const simulationRef = useRef<d3.Simulation<Node, undefined> | null>(null);
  const [links, setLinks] = useState<Array<{ source: Node; target: Node }>>([]);
  const [selectedUser, setSelectedUser] = useState<Node | null>(null);

  useEffect(() => {
    if (!svgRef.current || !gRef.current) return;

    // Set up SVG and zoom
    const svg = d3.select(svgRef.current);
    const g = d3.select(gRef.current);

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform.toString());
      });

    svg.call(zoom);

    // Initialize simulation if it doesn't exist
    if (!simulationRef.current) {
      simulationRef.current = d3.forceSimulation<Node>()
        .force('charge', d3.forceManyBody().strength(-2000))
        .force('collide', d3.forceCollide().radius(CARD_WIDTH))
        .force('center', d3.forceCenter(
          window.innerWidth / 2,
          window.innerHeight / 2
        ))
        .force('link', d3.forceLink([])
          .id(d => (d as Node).handle)
          .distance(CARD_WIDTH)
          .strength(0.1)
        )
        .on('tick', () => {
          setNodes(nodes => [...nodes]);
        });
    }

    // Update simulation nodes
    simulationRef.current.nodes(nodes);
    
    // Create links from follows
    const newLinks: Array<{ source: Node; target: Node }> = [];
    const handleMap = new Map(nodes.map(node => [node.handle, node]));

    nodes.forEach(sourceNode => {
      sourceNode.follows?.forEach(targetHandle => {
        const targetNode = handleMap.get(targetHandle);
        if (targetNode) {
          newLinks.push({ source: sourceNode, target: targetNode });
        }
      });
    });

    // Update simulation links
    simulationRef.current.force<d3.ForceLink<Node, any>>('link')?.links(newLinks);
    
    // Start simulation with lower alpha
    simulationRef.current.alpha(0.5).restart();

    // Update links state for rendering
    setLinks(newLinks);

    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [nodes.length, setNodes]);

  const calculateArrowPoints = (source: Node, target: Node) => {
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const angle = Math.atan2(dy, dx);

    // Calculate intersection with rectangle
    // Using the larger of width/height ratios to ensure we hit the edge
    const sourceRatio = Math.min(
      Math.abs(CARD_WIDTH / (2 * Math.cos(angle))),
      Math.abs(CARD_HEIGHT / (2 * Math.sin(angle)))
    );
    const targetRatio = Math.min(
      Math.abs(CARD_WIDTH / (2 * Math.cos(angle))),
      Math.abs(CARD_HEIGHT / (2 * Math.sin(angle)))
    );

    // Calculate the points where the arrow should start/end
    const sourceX = source.x + (sourceRatio * Math.cos(angle));
    const sourceY = source.y + (sourceRatio * Math.sin(angle));
    const targetX = target.x - (targetRatio * Math.cos(angle));
    const targetY = target.y - (targetRatio * Math.sin(angle));

    return { sourceX, sourceY, targetX, targetY };
  };

  return (
    <div className="w-full h-screen fixed inset-0 overflow-hidden bg-gray-900">
      <svg ref={svgRef} className="w-full h-full">
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" className="fill-gray-400" />
          </marker>
        </defs>
        
        <g ref={gRef}>
          {links.map((link, i) => {
            const { sourceX, sourceY, targetX, targetY } = calculateArrowPoints(link.source, link.target);
            
            return (
              <path
                key={i}
                d={`M ${sourceX} ${sourceY} L ${targetX} ${targetY}`}
                className="stroke-gray-400 stroke-2 fill-none"
                markerEnd="url(#arrowhead)"
              />
            );
          })}
          
          {nodes.map((node) => (
            <foreignObject
              key={node.handle}
              x={node.x - CARD_WIDTH / 2}
              y={node.y - CARD_HEIGHT / 2}
              width={CARD_WIDTH}
              height={CARD_HEIGHT}
              className="cursor-pointer"
              onClick={() => setSelectedUser(node)}
            >
              <UserCard profile={node} />
            </foreignObject>
          ))}
        </g>
      </svg>

      {/* User Panel */}
      {selectedUser && (
        <UserPanel 
          user={selectedUser} 
          onClose={() => setSelectedUser(null)} 
        />
      )}
    </div>
  );
} 