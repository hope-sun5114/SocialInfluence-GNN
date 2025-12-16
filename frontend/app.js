async function loadJSON(path) {
    const res = await fetch(path);
    return await res.json();
}

// ⭐ 1. Top-K 影响力排名
async function renderRanking() {
    const data = await loadJSON("data/ranking.json");

    const chart = echarts.init(document.getElementById("rankChart"));

    chart.setOption({
        title: { text: "节点影响力 Top-K 排名", left: "center" },
        tooltip: {},
        xAxis: { type: "value" },
        yAxis: { type: "category", data: data.names },
        series: [{
            type: "bar",
            data: data.scores,
            itemStyle: { color: "#4B7BEC" }
        }]
    });
}

// ⭐ 2. 传播图结构
async function renderGraph() {
    const graph = await loadJSON("data/graph.json");

    const chart = echarts.init(document.getElementById("graphChart"));

    chart.setOption({
        title: { text: "传播图结构", left: "center" },
        series: [{
            type: "graph",
            layout: "force",
            roam: true,
            data: graph.nodes,
            edges: graph.edges,
            force: { repulsion: 150 },
            label: { show: false },
            symbolSize: 10,
            lineStyle: { color: "#aaa" },
            emphasis: { focus: "adjacency" }
        }]
    });
}

// ⭐ 3. t-SNE
async function renderTSNE() {
    const tsne = await loadJSON("data/tsne.json");

    const chart = echarts.init(document.getElementById("tsneChart"));

    chart.setOption({
        title: { text: "t-SNE 节点嵌入可视化", left: "center" },
        xAxis: {},
        yAxis: {},
        series: [{
            type: "scatter",
            data: tsne.points,
            symbolSize: 8,
            itemStyle: { color: "#20BF6B" }
        }]
    });
}

renderRanking();
renderGraph();
renderTSNE();
