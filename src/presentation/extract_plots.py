import json
import sys

ipynb_path = '/Users/santiagoarellano/Documents/Projects/MidtermDataAnalysisProject/src/python/notebooks/Proyecto_Prueba_Tecnica_Arellano.ipynb'
html_path = '/Users/santiagoarellano/Documents/Projects/MidtermDataAnalysisProject/src/presentation/index.html'

with open(ipynb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

plots = {}

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'data' in out and 'application/vnd.plotly.v1+json' in out['data']:
                plotly_data = out['data']['application/vnd.plotly.v1+json']
                title = ''
                try:
                    title = plotly_data.get('layout', {}).get('title', {}).get('text', '')
                except:
                    pass
                if 'Conteo por hora' in title:
                    plots['slide-2'] = plotly_data
                elif 'Conteo por d' in title and 'technique' not in title:
                    # Not requested explicitly for a specific slide, maybe slide-2? Or maybe we map titles directly.
                    # Wait, let's map by title
                    plots['Conteo por día de la semana y evento'] = plotly_data
                elif 'Conteo por d' in title and 'technique' in title:
                    plots['slide-3'] = plotly_data
                elif 'Embudo' in title:
                    plots['slide-4'] = plotly_data
                elif 'Flujo' in title:
                    plots['slide-5'] = plotly_data

print("Found plots for slides:", list(plots.keys()))
