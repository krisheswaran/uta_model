'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from 'recharts';

interface Props {
  distribution: Record<string, number>;
}

const PRIMARY = '#D0BCFF';
const PRIMARY_DIM = 'rgba(208,188,255,0.4)';

export default function TacticBarChart({ distribution }: Props) {
  const data = Object.entries(distribution)
    .map(([tactic, count]) => ({ tactic, count }))
    .sort((a, b) => b.count - a.count);

  if (data.length === 0) {
    return (
      <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14, padding: 24 }}>
        No tactic distribution data available.
      </div>
    );
  }

  const maxCount = Math.max(...data.map((d) => d.count));

  return (
    <div style={{ width: '100%' }}>
      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 38)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 60, bottom: 8, left: 8 }}
        >
          <XAxis
            type="number"
            domain={[0, maxCount]}
            tick={{ fill: 'rgba(202,196,208,0.6)', fontSize: 11 }}
            axisLine={{ stroke: 'rgba(147,143,153,0.3)' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="tactic"
            width={160}
            tick={{ fill: 'var(--md-sys-color-on-surface)', fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: 'rgba(208,188,255,0.05)' }}
            contentStyle={{
              background: 'var(--md-sys-color-surface-container-highest)',
              border: '1px solid var(--md-sys-color-outline-variant)',
              borderRadius: 8,
              color: 'var(--md-sys-color-on-surface)',
              fontSize: 13,
            }}
            formatter={(value: number) => [`${value} beats`, 'Count']}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={entry.tactic}
                fill={index === 0 ? PRIMARY : PRIMARY_DIM}
                opacity={1 - index * (0.4 / Math.max(data.length - 1, 1))}
              />
            ))}
            <LabelList
              dataKey="count"
              position="right"
              style={{ fill: 'var(--md-sys-color-on-surface-variant)', fontSize: 12 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
