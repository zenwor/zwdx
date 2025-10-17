import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function JobProgressGraph({ progress }) {
  const epochData = useMemo(() => {
    const grouped = {};
    progress.forEach((step) => {
      if (!grouped[step.epoch] || step.batch > grouped[step.epoch].batch) {
        grouped[step.epoch] = {
          epoch: step.epoch,
          batch: step.batch,
          loss: Number(step.loss ?? 0),
          accuracy: Number(step.accuracy ?? 0),
        };
      }
    });
    return Object.values(grouped).sort((a, b) => a.epoch - b.epoch);
  }, [progress]);

  return (
    <div style={{ width: "100%", height: 400, marginTop: "20px" }}>
      <ResponsiveContainer>
        <LineChart
          data={epochData}
          margin={{ top: 20, right: 60, left: 60, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="epoch"
            label={{ value: "Epoch", position: "insideBottom", offset: -5 }}
          />

          {/* Left Y Axis */}
          <YAxis
            yAxisId="loss"
            label={{
              value: "Loss (Left Scale)",
              angle: -90,
              position: "insideLeft",
              dx: -10, 
              dy: 0,
              style: { textAnchor: "middle" },
            }}
            domain={['auto', 'auto']}
            allowDecimals
          />

          {/* Right Y Axis */}
          <YAxis
            yAxisId="accuracy"
            orientation="right"
            label={{
              value: "Accuracy %",
              angle: -90,
              position: "insideRight",
              dx: 10, 
              dy: 0,
              style: { textAnchor: "middle" },
            }}
            domain={[0, 100]}
          />

          <Tooltip />
          <Legend verticalAlign="top" height={36} />

          <Line
            yAxisId="loss"
            type="monotone"
            dataKey="loss"
            name="Loss (Left)"
            stroke="#ff7300"
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />

          <Line
            yAxisId="accuracy"
            type="monotone"
            dataKey="accuracy"
            name="Accuracy % (Right)"
            stroke="#387908"
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
