import json
import sys

ipynb_path = '/Users/santiagoarellano/Documents/Projects/MidtermDataAnalysisProject/src/python/notebooks/Proyecto_Prueba_Tecnica_Arellano.ipynb'
html_path = '/Users/santiagoarellano/Documents/Projects/MidtermDataAnalysisProject/src/presentation/index.html'

with open(ipynb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

brand_colors = ['#FF8C7A', '#2DD4BF', '#FBBF24', '#38BDF8', '#A78BFA', '#F472B6', '#34D399']

scripts = []
scripts.append("<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>")
scripts.append("<script>")

def process_plotly_data(plotly_data, slide_id):
    color_idx = 0
    for trace in plotly_data['data']:
        if 'marker' in trace:
            if isinstance(trace['marker'].get('color'), list):
                trace['marker']['color'] = [brand_colors[i % len(brand_colors)] for i in range(len(trace['marker']['color']))]
            else:
                trace['marker']['color'] = brand_colors[color_idx % len(brand_colors)]
                color_idx += 1
        if 'line' in trace:
            if not isinstance(trace['line'].get('color'), list):
                trace['line']['color'] = brand_colors[color_idx % len(brand_colors)]
    
    if 'layout' not in plotly_data:
        plotly_data['layout'] = {}
    
    plotly_data['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
    plotly_data['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
    if 'font' not in plotly_data['layout']:
        plotly_data['layout']['font'] = {}
    plotly_data['layout']['font']['color'] = '#F8FAFC'
    
    for axis in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
        if axis in plotly_data['layout']:
            plotly_data['layout'][axis]['gridcolor'] = 'rgba(255,255,255,0.1)'
            plotly_data['layout'][axis]['zerolinecolor'] = 'rgba(255,255,255,0.2)'
            if 'title' in plotly_data['layout'][axis]:
                if 'font' not in plotly_data['layout'][axis]['title']:
                    plotly_data['layout'][axis]['title']['font'] = {}
                plotly_data['layout'][axis]['title']['font']['color'] = '#94A3B8'
            
    div_id = f"plotly-{slide_id}"
    
    data_json = json.dumps(plotly_data['data'])
    layout_json = json.dumps(plotly_data['layout'])
    
    scripts.append(f"Plotly.newPlot('{div_id}', {data_json}, {layout_json}, {{responsive: true, displayModeBar: false}});")

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'data' in out and 'application/vnd.plotly.v1+json' in out['data']:
                plotly_data = out['data']['application/vnd.plotly.v1+json']
                try:
                    title = plotly_data.get('layout', {}).get('title', {}).get('text', '')
                except:
                    title = ''
                
                if 'Conteo por hora' in title:
                    process_plotly_data(plotly_data, 'slide-2')
                elif 'Conteo por d' in title and 'technique' in title:
                    process_plotly_data(plotly_data, 'slide-3')
                elif 'Embudo' in title:
                    process_plotly_data(plotly_data, 'slide-4')
                elif 'Flujo' in title:
                    process_plotly_data(plotly_data, 'slide-5')
                # Wait, slide 6 was Co-occurrence, which was matplotlib. The user asked for plotly visualizations. So slide 6 won't have one injected here.

scripts.append("</script>")

with open(html_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

if "https://cdn.plot.ly/plotly-2.27.0.min.js" not in html_content:
    html_content = html_content.replace('</body>', '\n'.join(scripts) + '\n</body>')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Injected Plotly scripts into index.html")
else:
    print("Plotly scripts already in index.html")
